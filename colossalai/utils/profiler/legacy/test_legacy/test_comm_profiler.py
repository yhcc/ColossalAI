import torch                


from multiprocessing import Pool, get_context
import torch.distributed as dist

from colossalai.utils.profiler.legacy.comm_profiler import CommProfiler
from colossalai.utils.profiler.legacy.prof_utils import ProfilerContext
from colossalai.communication.p2p import filling_ops_queue

def test_p2p_ops_case1(group, rank):
    a = torch.tensor([0.11111, 0.222222, 0.3333333], dtype=torch.float32, device=torch.device("cuda:{}".format(rank)))
    b = torch.as_tensor(a)
    if rank == 0:
        dist.send(a, 1, group)
        dist.send(b, 1, group)
    else:
        dist.recv(a, 0, group)
        dist.recv(b, 0, group)
    print("Case1 Rank:{} send/recv over".format(rank))

def test_p2p_ops_case2(group, rank):
    a = torch.tensor([0.11111, 0.222222, 0.3333333], dtype=torch.float32, device=torch.device("cuda:{}".format(rank)))
    b = torch.as_tensor(a)
    if rank == 0:
        dist.send(a, 1, group)
        dist.recv(b, 1, group)
    else:
        dist.recv(a, 0, group)
        dist.send(b, 0, group)
    print("Case2 Rank:{} send/recv over".format(rank))

def test_p2p_ops_case3(group, rank):
    a = torch.tensor([0.11111, 0.222222, 0.3333333], dtype=torch.float32, device=torch.device("cuda:{}".format(rank)))
    b = torch.as_tensor(a)
    c = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32, device=torch.device("cuda:{}".format(rank)))
    if rank == 0:
        dist.send(a, 1, group)
        print(f"{rank} send done", flush=True)
        dist.send(b, 1, group)
        print(f"{rank} send done", flush=True)
        torch.cuda.synchronize()
        print(f"{rank} sync done", flush=True)
        dist.all_gather_into_tensor(c, a, group, async_op=False)
        print(f"{rank} all_gather_into_tensor done", flush=True)
    else:
        dist.recv(a, 0, group)
        print(f"{rank} recv done", flush=True)
        dist.recv(b, 0, group)
        print(f"{rank} recv done", flush=True)
        torch.cuda.synchronize()
        print(f"{rank} sync done", flush=True)
        dist.all_gather_into_tensor(c, b, group, async_op=False)
        print(f"{rank} all_gather_into_tensor done", flush=True)
    print("Case3 Rank:{} send/recv over".format(rank))


def test_p2p_ops_case_async(group, rank):
    a = torch.tensor([0.11111, 0.222222, 0.3333333], dtype=torch.float32, device=torch.device("cuda:{}".format(rank)))
    b = torch.as_tensor(a)
    if rank == 0:
        handler1 = dist.isend(a, 1, group)
        print(f"{rank} isend done", flush=True)
        handler2 = dist.isend(b, 1, group)
        print(f"{rank} isend done", flush=True)
        handler1.wait()
        handler2.wait()
    else:
        handler1 = dist.irecv(a, 0, group)
        print(f"{rank} irecv done", flush=True)
        handler2 = dist.irecv(b, 0, group)
        print(f"{rank} irecv done", flush=True)
        handler1.wait()
        handler2.wait()
    print("Case async Rank:{} send/recv over".format(rank))

def test_torch_batch_isend_irecv(group, rank):
    a = torch.tensor([0.11111, 0.222222, 0.3333333], dtype=torch.float32, device=torch.device("cuda:{}".format(rank)))
    b = torch.as_tensor(a)
    ops = []
    if rank == 0:
        filling_ops_queue(a, dist.isend, 1, ops)
        filling_ops_queue(b, dist.isend, 1, ops)
        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
    else:
        filling_ops_queue(a, dist.irecv, 0, ops)
        filling_ops_queue(b, dist.irecv, 0, ops)
        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

def profiling_engine(rank):
    pg = dist.init_process_group("nccl", rank=rank, world_size=2, init_method="tcp://localhost:12398")
    prof = ProfilerContext([CommProfiler()])
    dist.barrier()
    print("begin test 1", flush=True)
    with prof:
        test_p2p_ops_case1(pg, rank)
    prof.show()
    prof.reset()
    
    dist.barrier()
    print("begin test 2", flush=True)
    with prof:
        test_p2p_ops_case2(pg, rank)
    prof.show()
    prof.reset()
    
    dist.barrier()
    print("begin test 3", flush=True)
    with prof:
        test_p2p_ops_case3(pg, rank)
    prof.show()
    prof.reset()

    dist.barrier()
    print("begin test 4 async", flush=True)
    with prof:
        test_p2p_ops_case_async(pg, rank)
    prof.show()
    prof.reset()
    
    dist.barrier()
    print("begin test 5 batch_isend_irecv", flush=True)
    with prof:
        test_torch_batch_isend_irecv(pg, rank)
    prof.show()
    prof.reset()


if __name__ == "__main__":
    ctx = get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        pool.map(profiling_engine, range(2))
        pool.close()
        pool.join()

