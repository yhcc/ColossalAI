
from colossalai.logging import get_dist_logger


def partition_uniform_with_embed(num_items, pipeline_parallel_size, num_chunks):
    """
    在partition的时候，会额外考虑到第一部分和最后一部分需要有embedding，所以少分配一层

    """
    num_items += 2  # 额外考虑 embedding 和 head 那两层
    assert num_items % num_chunks == 0, \
        "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"
    # num_items=6, pipeline_parallel_size=4, num_chunks=2
    logger = get_dist_logger()
    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        # left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        end = partition_items % pipeline_parallel_size
        start = -1
        if end<=pipeline_parallel_size-2:  # 如果剩余小于中间层数，尽量分配给中间层
            start = 0
            end += 1

        if chunk_size == 0:
            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (start < p < end)
            parts[p].append((st, base_idx))

    # 整体都要shift
    real_parts = []
    for idx, _parts in enumerate(parts):
        _real_parts = []
        for _idx, (s, e) in enumerate(_parts):
            s -= 1  # 都往前shift一个embedding
            e -= 1 
            if s<0:
                s = 0
            if e==num_items-1:  # 最后的head需要多减掉一位
                e -= 1
            if e-s>0:
                _real_parts.append((s, e))
            
        real_parts.append(_real_parts)

    # num_chunks=1 [[(star, end)], [(start, end)]...] 前闭后开
    # num_chunks=2 [[(star, end), (start, end)], [(start, end), (start, end)]...] 前闭后开
    # to make sure not wrong, add an assert
    indexes = []
    for _parts in real_parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items-2))), (indexes, num_items)  # should have the same indexes as expected
    return real_parts


def partition_without_last_head(num_items, pipeline_parallel_size, num_chunks):
    """
    在partition的时候，会额外考虑到第一部分和最后一部分需要有embedding，所以少分配一层

    """
    pipeline_parallel_size -= 1  # 最后一层只给 head 用
    assert num_chunks==1, "Currently only 1 chunk is allowed, whether it supports more chunks should be considered."
    assert num_items % num_chunks == 0, \
        "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"
    # num_items=6, pipeline_parallel_size=4, num_chunks=2
    logger = get_dist_logger()
    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        # left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        end = partition_items % pipeline_parallel_size
        start = -1
        if end<=pipeline_parallel_size-2:  # 如果剩余小于中间层数，尽量分配给中间层
            start = 0
            end += 1

        if chunk_size == 0:
            logger.warning("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (start < p < end)
            parts[p].append((st, base_idx))

    # num_chunks=1 [[(star, end)], [(start, end)]...] 前闭后开
    # num_chunks=2 [[(star, end), (start, end)], [(start, end), (start, end)]...] 前闭后开
    # to make sure not wrong, add an assert
    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected
    
    parts.append([])
    
    return parts



if __name__ == '__main__':
    # print(partition_uniform_with_embed(12, 4, 1))
    print(partition_without_last_head(12, 2, 1))
