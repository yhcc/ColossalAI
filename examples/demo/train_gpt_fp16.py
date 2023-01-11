
# torchrun --standalone --nproc_per_node 8 --nnodes 1 train_gpt_fp16.py --from_torch --config configs/demo_config.py 
# srun -p llm -n8 --ntasks-per-node 8 --gres=gpu:8 --job-name hello --kill-on-bad-exit=1 python train_gpt_fp16.py  --config configs/demo_config.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai import nn as col_nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from model.pipeline_gpt1d import GPT2_exlarge_pipeline_1D
import psutil
from tqdm import tqdm
from colossalai import nn as col_nn
from torch.utils.data import Dataset
from colossalai.utils import is_dp_rank_0, is_tp_rank_0, is_no_pp_or_last_stage

from model.utils import assert_current_device_empty
from colossalai.zero.sharded_optim.low_level_optim import LowLevelZeroOptimizer
from colossalai.amp.naive_amp import NaiveAMPModel

from rich.console import Console
logger = get_dist_logger()
logger._logger.handlers[0].console = Console(width=200)  # hack 一下logger
# import logging
# formatter = logging.Formatter('%(message)s')
# logger._logger.handlers[0].setFormatter(formatter)


# from dataset.webtext import WebtextDataset
def get_master_node():
    import subprocess
    result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
    result = result.decode('utf8').strip()
    return result


def calc_local_model_size(model:torch.nn.Module, with_embed=True):
    numel_per_device = 0
    for name, p in model.named_parameters():
        if not with_embed and 'embedding' in name:
            continue
        numel_per_device += p.numel()
    return numel_per_device


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2

def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'



class GPTLMLoss(nn.Module):

    def __init__(self, parallel_output=True):
        super().__init__()
        if parallel_output:
            self.loss_fn = col_nn.CrossEntropyLoss(reduction=True)  # 这个地方的loss和VocabParallelClassifier1D初始化的gather_output是绑定的
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')  # 这里由于在model中设置了输出会gather output，所以使用普通的 loss

    def forward(self, logits, labels, **kwargs):
        shift_logits = logits[..., 1:-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 2:].contiguous().view(-1)
        loss = self.loss_fn(shift_logits, shift_labels).sum()
        return loss

def calculate_acc(logits, labels):
    shift_logits = logits[..., 1:-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[..., 2:].contiguous().view(-1)
    pred_shift = 0
    if gpc.config.model['parallel_output']:
        process_group = gpc.get_group(ParallelMode.PARALLEL_1D)
        rank = torch.distributed.get_rank(process_group)
        pred_shift = rank*logits.shape[-1]  # 根据当前的rank有一个shift，因为logits是被切分掉了

    acc = (shift_labels == (shift_logits.argmax(dim=-1)+pred_shift)).sum()
    if gpc.config.model['parallel_output']:
        torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM, group=process_group)

    mask = shift_labels.ne(-100)
    acc = acc/mask.sum()

    return acc.item()
    
#############hyper
# num_layers = 48

#############hyper

import numpy as np

def get_megatron_flops(elapsed_time_per_iter):
    checkpoint_activations_factor = 4 if gpc.config.model.checkpoint else 3
    seq_len = gpc.config.SEQ_LEN
    hidden_size = gpc.config.model.hidden_size
    num_layers = gpc.config.model.num_layers
    vocab_size = gpc.config.model.vocab_size
    batch_size = gpc.config.BATCH_SIZE * gpc.get_world_size(ParallelMode.DATA)
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * gpc.config.model.num_layers * (gpc.config.model.hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    tflops = flops_per_iteration / (elapsed_time_per_iter *gpc.get_world_size(ParallelMode.GLOBAL) * (10**12))
    return tflops


class RandomDataset(Dataset):
    def __init__(self, num_samples=10000, pad=True) -> None:
        super().__init__()
        rng = np.random.RandomState(gpc.get_local_rank(ParallelMode.DATA))
        max_num = rng.randint(1, 30, size=(num_samples, ))
        rep_num = rng.randint(100, 200, size=(num_samples, ))
        data = []
        for n, r in zip(max_num, rep_num):
            d = [n, r] + list(range(n))*r
            # d = [n, r] + [n]*r
            d = d[:1024]
            data.append(d)
        self.data = data
        self.pad = pad

    def __getitem__(self, index):
        d = self.data[index]
        if self.pad:
            pad_d = d + [0]*(gpc.config.SEQ_LEN-len(d))
            label = d + [-100]*(gpc.config.SEQ_LEN-len(d))
            input_ids = np.array(pad_d, dtype=int)
        else:
            label = d 
            input_ids = np.array(d, dtype=int)
        # return {'input_ids': np.arange(gpc.config.SEQ_LEN).astype(np.int64)}, np.arange(gpc.config.SEQ_LEN).astype(np.int64)
        return {'input_ids': input_ids}, np.array(label, dtype=int)

    def __len__(self):
        return len(self.data)


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    parser.add_argument(
        "--flop_log_path",
        type=str,
        default=None,
        help=
        "Where to save flop info",
    )

    args = parser.parse_args()

    disable_existing_loggers()
    logger = get_dist_logger()

    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(config=args.config, host=get_master_node(), port=7777, seed=42)

    global_local_rank = colossalai.core.global_context.get_local_rank(ParallelMode.GLOBAL)

    logger.info('Build data loader', ranks=[0])
    logger.info(args, ranks=[0])

    train_ds = RandomDataset(num_samples=gpc.config.NUM_SAMPLES, pad=hasattr(gpc.config, 'TENSOR_SHAPE'))

    def colatte_fn(batch):
        xs, ys = [], []
        for x, y in batch:
            xs.append(torch.as_tensor(x['input_ids']))
            ys.append(torch.as_tensor(y))
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        return {'input_ids': xs}, ys

    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=413,
                                            batch_size=gpc.config.BATCH_SIZE,
                                            pin_memory=False,
                                            shuffle=True,
                                            drop_last=True,
                                            collate_fn=colatte_fn)

    model_config = gpc.config.model
    model_config.update(dict(dtype=torch.float))

    model = GPT2_exlarge_pipeline_1D(**model_config)

    numel = calc_local_model_size(model, with_embed=False)
    logger.info(f"The number of parameters without embedding are {numel}.", ranks=[0])
    numel = calc_local_model_size(model)
    logger.info(f"The number of parameters are {numel}.", ranks=[0])

    total_param = torch.tensor([numel]).cuda()
    torch.distributed.all_reduce(total_param,
                                 op=torch.distributed.ReduceOp.SUM, group=gpc.get_group(ParallelMode.MODEL))
    logger.error(f"The number of parameters are {total_param.item()}.", ranks=[0])
    # 312是Palm文章中汇报的A100的速度
    R_palm_in_TF = 312*gpc.get_world_size(ParallelMode.GLOBAL)/((6*total_param.item() + 12*gpc.config.model['num_layers']*gpc.config.model['hidden_size']*gpc.config.SEQ_LEN)/10**12)
    logger.error(f"PaLM optimal Token/s:{R_palm_in_TF}, Global batch size:{gpc.config.BATCH_SIZE*gpc.get_world_size(ParallelMode.DATA)}",
    ranks=[0])

    # exit()
    criterion = GPTLMLoss(parallel_output=gpc.config.model['parallel_output'])
    model = NaiveAMPModel(model=model, output_to_fp32 = is_no_pp_or_last_stage())  # 这里的output_to_fp32一定需要格外注意，不然会在pipe间卡住

    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.optimizer['lr'], 
                                    weight_decay=gpc.config.optimizer['weight_decay'])      # 这里就不需要使用什么HybraOptimizer了，因为不存在multi_tensor更新了
    optimizer = LowLevelZeroOptimizer(optimizer, overlap_communication=False,           # 需要在model转为half之后，否则内部的获取会有问题
                                             partition_grad=False,
                                             verbose=True, 
                                             clip_grad_norm=gpc.config.clip_grad_norm,
                                             initial_scale=gpc.config.fp16['initial_scale'], 
                                             min_scale=gpc.config.fp16['min_scale'],
                                             growth_interval=gpc.config.fp16['growth_interval'],
                                             max_scale=None)

    gpc.config.pop('fp16')  # 不要使用fp16了

    logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    total_steps = len(train_dataloader) * gpc.config.NUM_EPOCHS

    lr_scheduler = LinearWarmupLR(
        optimizer, total_steps=total_steps, warmup_steps=int(total_steps*0.1))
    gpc.config.gradient_handler = [dict(type='PipelineSharedModuleGradientHandler')]  # 防止initialize加入ddp的gradient handler和lowleveoptimizer冲突；同时需要为pipeline特别考虑
    engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion=criterion,
                                                                      train_dataloader=train_dataloader,
                                                                      lr_scheduler=lr_scheduler)

    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    engine.train()
    enable = is_dp_rank_0() and is_tp_rank_0() and is_no_pp_or_last_stage()
    pbar = tqdm(total=total_steps, leave=False, disable=True, position=0)
    start = 0

    from functools import partial
    import time
    def get_tflops(model_numel, batch_size, seq_len, step_time):
        return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)
    get_tflops_func = partial(get_tflops, numel, gpc.config.BATCH_SIZE, gpc.config.SEQ_LEN)

    # for epoch in range(gpc.config.NUM_EPOCHS):
    data_iter = iter(train_dataloader)
    flops_lst = []

    total_steps = 10**9

    for step in range(start, start+len(train_dataloader)):
        try:
            start_time = time.time()
            engine.zero_grad()
            logits, label, loss = engine.execute_schedule(
                data_iter,
                forward_only=False,
                return_loss=True,
                return_output_label=False,
            )
            # if enable:
            #     for name, param in model.named_parameters():
            #         pbar.write(f'{step}, {name}, {type(param.grad)}')
            #         break
            success_update, grad_norm = engine.step()
            if success_update:  # 只有update成功才更新，之后这里顺便需要修改一下记录对应的sample数量，这个才是有效的sample数量
                lr_scheduler.step()
            acc = 0
            if logits is not None:
                acc = calculate_acc(logits, label)
            if enable and success_update:
                lr = optimizer.param_groups[0]['lr']
                if gpc.config.get('gradient_accumulation', None) is not None:
                    scaler = engine.optimizer.optim.optim.grad_scaler._scale.item()
                elif hasattr(engine.optimizer, 'grad_scaler'):
                    scaler = engine.optimizer.grad_scaler._scale.item()
                elif hasattr(engine.optimizer.optim, 'grad_scaler'):
                    scaler = engine.optimizer.optim.grad_scaler._scale.item()
                else:
                    scaler = -1
                flops = get_tflops_func(time.time() - start_time)
                mflops = get_megatron_flops(time.time() - start_time)
                token_per_second = gpc.config.BATCH_SIZE*gpc.get_world_size(ParallelMode.DATA)*gpc.config.SEQ_LEN/(time.time() - start_time)
                token_per_second = round(token_per_second/gpc.get_world_size(ParallelMode.GLOBAL), 2)
                line = f'Loss:{round(loss.item(), 6)}, acc:{round(acc, 6)}, lr:{lr}, scaler:{scaler}, norm:{grad_norm}, mflops:{mflops}, t/s/gpu:{token_per_second}'
                pbar.set_postfix_str(line)
                logger.info(line)
                flops_lst.append(mflops)
                pbar.update()
            start += 1
            if step > total_steps:
                break
        except Exception as e:
            logger.error(f"Error in rank:{gpc.get_global_rank()}!!!!")
            raise e
assert_current_device_empty()
main()


