from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
import torch
import os
from petrel_client.client import Client
import io
from colossalai.logging import get_dist_logger
from torch.distributed import ReduceOp
from colossalai.utils.checkpointing import broadcast_model


def get_dp_size():
    """
    获取data parallel的数量，没有的话就是1
    """
    try:
        return gpc.get_world_size(ParallelMode.DATA)
    except KeyError:
        return 1

def get_tp_size():
    """
    获取tensor parallel的数量，没有的话就是1
    """
    try:
        return gpc.get_world_size(ParallelMode.TENSOR)
    except KeyError:
        return 1

def get_pp_size():
    """
    获取pipeline parallel的数量，没有的话就是1
    """
    try:
        return gpc.get_world_size(ParallelMode.PIPELINE)
    except KeyError:
        return 1


def get_dp_rank():
    try:
        return gpc.get_local_rank(ParallelMode.DATA)
    except KeyError:
        return 0


def get_tp_rank():
    try:
        return gpc.get_local_rank(ParallelMode.TENSOR)
    except KeyError:
        return 0


def get_pp_rank():
    try:
        return gpc.get_local_rank(ParallelMode.PIPELINE)
    except KeyError:
        return 0


def load_model_checkpoint(model, folder):
    """
    model: 加载权重的模型
    folder: 从哪个folder读取，这个folder下应该有类似于model_{pp_rank}.pt的文件。
    当前仅支持pipeline parallel完全一致的情况
    """
    max_pp = 0
    if folder.startswith('s3://'):
        client = Client()
        fns = list(client.list(folder))
    else:
        assert os.path.exists(folder)
        fns = os.listdir(folder)
    for fn in fns:
        if fn.startswith('model_'):
            _, pp = os.path.splitext(fn)[0].split('_')
            max_pp = max(max_pp, int(pp))
    
    assert get_pp_size() == max_pp + 1, f"The weights are save for {max_pp+1} pipelines, while current has {get_pp_size()} pipelines" 

    pp_rank = get_pp_rank()

    should_load_name = f'model_{pp_rank}.pt'
    fp = os.path.join(folder, should_load_name)
    if get_tp_rank() == 0:
        if folder.startswith('s3://'):
            buffer = client.get(fp)
            states = torch.load(buffer, map_location='cpu')
        else:
            with open(fp, 'rb') as f:
                states = torch.load(f, map_location='cpu')
    else:
        states = {}
    
    # TODO 暂时通过删除状态中第一个'.model'苟住
    states = {key[:key.index('.')]+key[key.index('.', key.index('.')+1):]:value for key,value in states.items()}

    model.load_state_dict(states)
    broadcast_model(model)

    torch.distributed.barrier()


def save_model_checkpoint(model, folder):
    """
    model: 需要保存的模型
    folder: 保存在哪个folder下，会在这个folder下生成类似于model_{pp_rank}.pt的文件，同时为了性能，整个存储过程是分布式的，即会尽量安排到不同的data parallel机器上

    """
    if folder.startswith('s3://'):
        client = Client()
        assert client.contains(folder)
    else:
        assert os.path.exists(folder)

    dp_size = get_dp_size() # 完全一致的model
    pp_size = get_pp_size()
    
    should_save_rank_tuple = set()

    for _pp in range(pp_size):
        should_save_rank_tuple.add((_pp%dp_size, _pp))

    cur_rank_tuple = (get_dp_rank(), get_pp_rank())
    states = model.state_dict()  # 会自动将tensor并行的汇聚到一起
    logger = get_dist_logger()
    logger.error(should_save_rank_tuple, ranks=[0])
    if cur_rank_tuple in should_save_rank_tuple:
        logger.info(f"Saving from global rank: {gpc.get_global_rank()}")
        fn = f'model_{cur_rank_tuple[1]}.pt'
        fp = os.path.join(folder, fn)
        if fp.startswith('s3://'):
            buffer = io.BytesIO()
            torch.save(states, buffer)
            client.put(fp, buffer.getvalue())
        else:
            torch.save(states, fp)
    torch.distributed.barrier()


def assert_current_device_empty():
    from pynvml.smi import nvidia_smi
    import socket
    nvsmi = nvidia_smi.getInstance()
    result = nvsmi.DeviceQuery('memory.used')['gpu']
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        devices = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        if 'SLURM_LOCALID' in os.environ:
            device_id = devices[int(os.environ['SLURM_LOCALID'])]
            cur_device_used = result[device_id]['fb_memory_usage']['used']

            if cur_device_used>100:
                raise RuntimeError(f"The `{device_id}` device of node `{socket.gethostname()}` is occupied with `{cur_device_used}` Mib before running.")



def save_flop_record(flop_log_path, flops_lst, palm_flops, token_per_sec, start_time):
    # 记录flops
    if flop_log_path is not None and gpc.get_local_rank(ParallelMode.DATA)==0 and gpc.is_last_rank(ParallelMode.PIPELINE) and gpc.get_local_rank(ParallelMode.PARALLEL_1D)==0 and len(flops_lst)!=0:
        path = flop_log_path
        from datetime import datetime
        import numpy as np
        import json
        import socket

        now = str(datetime.now())
        configs = dict(**gpc.config.model)
        configs.update(**gpc.config.parallel)
        configs['time'] = now
        configs['batch_size'] = gpc.config.BATCH_SIZE
        configs['length'] = gpc.config.SEQ_LEN
        mean_flops = float(np.mean(flops_lst))
        median_flops = float(np.median(flops_lst))
        configs['Node'] = socket.gethostname()
        configs['slurm_nodes'] = os.environ['SLURM_NODELIST']
        configs['world_size'] = gpc.get_world_size(ParallelMode.GLOBAL)
        configs['start_time'] = start_time
        configs['median_flops'] = round(median_flops, 2)
        configs['flops'] = flops_lst
        configs['palm_flops'] = palm_flops
        configs['token_per_sec'] = token_per_sec
        configs['mean_flops'] = round(mean_flops, 2)
        configs.pop('dtype')
        with open(path, 'a', encoding='utf8') as f:
            f.write(json.dumps(configs)+'\n')

    torch.distributed.barrier()


def notify_flop_record(flop_log_path, flops_lst, start_time):
    # 表示开始记录flops，用于判定有些没有正确运行的结果
    if flop_log_path is not None and gpc.get_local_rank(ParallelMode.DATA)==0 and gpc.is_last_rank(ParallelMode.PIPELINE) and gpc.get_local_rank(ParallelMode.PARALLEL_1D)==0:
        path = flop_log_path
        from datetime import datetime
        import numpy as np
        import json
        import socket

        now = str(datetime.now())
        configs = dict(**gpc.config.model)
        configs.update(**gpc.config.parallel)
        configs['time'] = now
        configs['batch_size'] = gpc.config.BATCH_SIZE
        configs['length'] = gpc.config.SEQ_LEN
        configs['Node'] = socket.gethostname()
        configs['slurm_nodes'] = os.environ['SLURM_NODELIST']
        configs['start_time'] = start_time
        configs.pop('dtype')
        with open(path, 'a', encoding='utf8') as f:
            f.write(json.dumps(configs)+'\n')

    torch.distributed.barrier()