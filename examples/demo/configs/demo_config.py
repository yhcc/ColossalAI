
# 这个是单组机器的 batch大小
NUM_SAMPLES = 10000
BATCH_SIZE = 120
NUM_EPOCHS = 10000  # 修改这个调整epoch数量
SEQ_LEN = 2048
NUM_MICRO_BATCHES = 120
HIDDEN_SIZE = 12288
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)


optimizer = dict(
    lr=0.0001,
    weight_decay=1e-2,
)

model = dict(
    checkpoint=True,
    num_chunks=1,  # 每张卡需要forward多少次。
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=96,
    num_layers=8,  # 没有包含 embedding 和 head
    embed_split_hidden=False,
    vocab_size=50256,
    embed_grad_scale=0.1,  # 清华的embedding放缩技巧，如果为1的话，不放缩
    parallel_output=True  # 最后的输出是否需要gather起来，如果不gather的话，每个tensor parallel获取的就是自己对应的结果
)

fp16 = dict(
    # below are the default values
    # log_num_zeros_in_grad=False,
    initial_scale=2 ** 16,
    min_scale=1,
    growth_factor=2,
    backoff_factor=0.5,
    growth_interval=1000,
    hysteresis=2,

    # sync_buffer=False,  # 无法对model进行操作
    # output_to_fp32=False
)
clip_grad_norm = 1.0
use_deep_fused = True
# gradient_accumulation = 3

# pipeline parallel: modify integer value for the number of pipeline stages
# tensor parallel: modify size to set the tensor parallel size, usually the number of GPUs per node
# for the current model implementation, mode can only be 1D or None
parallel = dict(
    pipeline=2,  # pipeline是指多少张卡来进行流水线并行
    tensor=dict(size=2, mode='1d'), # for the current model implementation, mode can only be 1D or None
)
