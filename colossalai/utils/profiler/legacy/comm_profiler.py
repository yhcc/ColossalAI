import inspect
from pathlib import Path
from functools import partial
import torch
import time
from torch.autograd.profiler import profile
# from torch.profiler import profile

import torch.distributed as dist
from torch.distributed import ReduceOp, get_backend
from colossalai.utils import get_current_device
from .prof_utils import BaseProfiler, _format_time, _format_memory, _format_bandwidth
from typing import List, Optional
from torch.autograd.profiler_util import FunctionEvent
from dataclasses import dataclass
from colossalai.logging import get_dist_logger
from colossalai.utils.profiler.legacy.tb_writer import get_global_counter, get_tb_manager
from torch.utils.tensorboard import SummaryWriter

# Find longest prefix substr of given strings.
def longestCommonPrefix(strs : List[str]):
    res = ""
    for tmp in zip(*strs):
        tmp_set = set(tmp)
        if len(tmp_set) == 1:
            res += tmp[0]
        else:
            break
    return res


def _get_code_location(depth: int):
    ret = []
    length = min(len(inspect.stack()), depth + 1)
    for i in range(3, length):
        upper_frame = inspect.stack()[i]
        function_name = inspect.stack()[i - 1].function
        ret.append(upper_frame.filename)
        ret.append('(')
        ret.append(str(upper_frame.lineno))
        ret.append('): ')
        ret.append(function_name)
        if i != length - 1:
            ret.append('\n')

    return ''.join(ret)


torch_all_reduce = dist.all_reduce
torch_all_gather = dist.all_gather
torch_all_gather_into_tensor = dist.all_gather_into_tensor
torch_reduce_scatter = dist.reduce_scatter
torch_broadcast = dist.broadcast
torch_reduce = dist.reduce
torch_send = dist.send
torch_recv = dist.recv
torch_isend = dist.isend
torch_irecv = dist.irecv
torch_batch_isend_irecv = dist.batch_isend_irecv

# @dataclass
class CommEvent:
    """Communication Event. Used for communication time and communication
    volume recording.
    """

    def __init__(self):
        self.self_count = 0
        self.self_comm_vol = 0.0
        self.self_cuda_time = 0.0
        self.max_self_cuda_time = 0
        self.min_self_cuda_time = float('inf')

        self.copy_cuda_time = 0.0
        self.h2d_time = 0.0
        self.h2d_count = 0
        self.d2h_time = 0.0
        self.d2h_count = 0

    def add(self, rhs):
        self.self_count += rhs.self_count
        self.self_comm_vol += rhs.self_comm_vol
        self.self_cuda_time += rhs.self_cuda_time

        self.max_self_cuda_time = max(self.max_self_cuda_time, rhs.self_cuda_time)
        self.min_self_cuda_time = min(self.min_self_cuda_time, rhs.self_cuda_time)

        self.copy_cuda_time = rhs.copy_cuda_time
        self.h2d_count = rhs.h2d_count
        self.h2d_time = rhs.h2d_time
        self.d2h_count = rhs.d2h_count
        self.d2h_time = rhs.d2h_time

    def __str__(self) -> str:
        return "self_count:{}, self_comm_vol:{}, self_cuda_time:{}, copy_cuda_time:{}".format(
            self.self_count, self.self_comm_vol, self.self_cuda_time, self.copy_cuda_time)


class CommProfiler(BaseProfiler):
    """Communication profiler. Records all communication events.
    """
    PCIE_KN_SET = {"Memcpy HtoD", "Memcpy DtoH", "aten::copy_"}

    def __init__(self,
                 depth: int = 0,
                 total_count: int = 0,
                 total_comm_vol: float = 0,
                 total_cuda_time: int = 0,
                 writer: SummaryWriter = None,
                 use_profile: bool = False):
        super().__init__(profiler_name="Collective_Communication", priority=0)
        self.depth = 3 + depth
        self.total_count = total_count
        self.total_comm_vol = total_comm_vol
        self.total_cuda_time = total_cuda_time

        self.total_copy_cuda_time = 0
        self.total_h2d_count = 0
        self.total_h2d_time = 0
        self.total_d2h_count = 0
        self.total_d2h_time = 0

        self.ops_record = dict()
        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None
        self.warn_flag = False

        self.logger = get_dist_logger()
        self.writer = writer
        self.pending_p2p = None
        self.send_op_count = 0
        self.recv_op_count = 0
        self.aysnc_p2p = 0
        
        self.use_profile = use_profile

    def reset(self):
        self.total_count = 0
        self.total_comm_vol = 0
        self.total_cuda_time = 0

        self.ops_record = dict()
        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None
        self.warn_flag = False

        self.total_copy_cuda_time = 0
        self.total_h2d_count = 0
        self.total_h2d_time = 0
        self.total_d2h_count = 0
        self.total_d2h_time = 0

        self.pending_p2p = None
        self.send_op_count = 0
        self.recv_op_count = 0
        self.aysnc_p2p = 0
    
    def add_total(self, event):
        self.total_count += event.self_count
        self.total_comm_vol += event.self_comm_vol
        self.total_cuda_time += event.self_cuda_time

        self.total_copy_cuda_time += event.copy_cuda_time
        self.total_h2d_count += event.h2d_count
        self.total_h2d_time = +event.h2d_time
        self.total_d2h_count += event.d2h_count
        self.total_d2h_time += event.d2h_time
            
    def enable(self):
        dist.all_reduce = partial(all_reduce, profiler=self)
        dist.all_gather = partial(all_gather, profiler=self)
        dist.all_gather_into_tensor = partial(all_gather_into_tensor, profiler=self)
        dist.reduce_scatter = partial(reduce_scatter, profiler=self)
        dist.broadcast = partial(broadcast, profiler=self)
        dist.reduce = partial(reduce, profiler=self)
        dist.send = partial(send, profiler=self)
        dist.recv = partial(recv, profiler=self)
        # dist.isend = partial(isend, profiler=self)
        # dist.irecv = partial(irecv, profiler=self)
        dist.batch_isend_irecv = partial(batch_isend_irecv, profiler=self)

    def disable(self):
        # When exit, we need to check whether there are ops without profiling. 
        if self.pending_metadata and self.pending_op:
            self.close_profiler()
        elif self.pending_p2p:
            self.close_p2p_profiler()

        dist.all_reduce = torch_all_reduce
        dist.all_gather = torch_all_gather
        dist.all_gather_into_tensor = torch_all_gather_into_tensor
        dist.reduce_scatter = torch_reduce_scatter
        dist.broadcast = torch_broadcast
        dist.reduce = torch_reduce
        dist.send = torch_send
        dist.recv = torch_recv
        # dist.isend = torch_isend
        # dist.irecv = torch_irecv
        dist.batch_isend_irecv = torch_batch_isend_irecv
    
    def add_record(self, prefix, event : CommEvent):
        if event.self_count > 0 :
            if prefix in self.ops_record:
                self.ops_record[prefix].add(event)
            else:
                self.ops_record[prefix] = event

            self.step_to_tensorboard(prefix, event.self_cuda_time, self.ops_record[prefix].self_count)
    
    def get_p2p_index(self):
        return self.send_op_count + self.recv_op_count

    # def check_can_end_p2p2_profile(self):
    #     if self.aysnc_p2p == 0:
    #         if self.send_op_count == self.recv_op_count:

    def find_frist_init_p2p_op(self, comm_vol, starce, is_send, op):
        if self.pending_p2p:
            send_vol, recv_vol, prefix, start_time, op_list = self.pending_p2p
            if is_send:
                send_vol += comm_vol
            else:
                recv_vol += comm_vol
            starce = longestCommonPrefix([prefix, starce])
        else:
            start_time = time.time()
            op_list = []
            if is_send:
                send_vol = comm_vol
                recv_vol = 0
            else:
                recv_vol = comm_vol
                send_vol = 0

        if op:
            op_list.append(op)
            self.aysnc_p2p += 1

        self.pending_p2p = (send_vol, recv_vol, starce, start_time, op_list)
        
        if is_send:
            self.send_op_count += 1
        else:
            self.recv_op_count += 1

        if self.send_op_count + self.recv_op_count == 1:
            self.activate_p2p_profiler()

    def step_to_tensorboard(self, kn, time, count):
        # Watch out tensorboard webpage overload
        if self.writer:
            self.writer.add_scalar(f'Collective Communication/{kn}', time, count)
            # self.writer.add_scalar(f'Collective Communication/step', count, get_global_counter())

    def to_tensorboard(self, writer=None):
        if self.writer:
            self.writer.add_text(tag="Collective Communication", text_string=self.result_str())

    def to_file(self, filename: Path):
        with open(filename, "w") as f:
            f.write(self.result_str())

    def show(self):
        print(self.result_str())

    def result_str(self, sep: str = "\n"):
        res = []

        def append(s: str = None):
            if s is not None:
                res.append(s)
            res.append(sep)

        if self.warn_flag:
            append("Warnning: there exists multiple communication operations in the same time. As a result, "
                   "the profiling result is not accurate.")

        if self.total_cuda_time == 0:
            return "No collective communication has been called yet!"

        append("Collective communication profiling result:")
        append("total cuda time: {}".format(_format_time(self.total_cuda_time)))
        append("average bandwidth: {}".format(_format_bandwidth(self.total_comm_vol, self.total_cuda_time)))
        append("total number of calls: {}".format(self.total_count))
        append("time of data transmission (CPU -> GPU): {}".format(_format_time(self.total_h2d_time)))
        append("number of transmission (CPU -> GPU): {}".format(self.total_h2d_count))
        append("time of data transmission (GPU -> CPU): {}".format(_format_time(self.total_d2h_time)))
        append("number of transmission (GPU -> CPU): {}".format(self.total_d2h_count))

        append("All events:")

        seperation = '-' * 74
        row_format = '{:^10}' + '{:^12}' * 2 + '{:^16}' + '{:^12}' * 3

        append(seperation)
        append(
            row_format.format('Location', 'GPU time', 'Percentage', 'Comm volume', 'Bandwidth', 'PCIe BW',
                              'Num of calls'))
        append(seperation)

        show_list = sorted(self.ops_record.items(), key=lambda kv: -kv[1].self_cuda_time)
        for location, event in show_list:
            event: CommEvent
            append(location)
            append(
                row_format.format('', _format_time(event.self_cuda_time),
                                  '{:.1f}%'.format(event.self_cuda_time / self.total_cuda_time * 100.0),
                                  _format_memory(event.self_comm_vol),
                                  _format_bandwidth(event.self_comm_vol, event.self_cuda_time),
                                  _format_bandwidth(event.self_comm_vol, event.copy_cuda_time), event.self_count))
            append()

        return ''.join(res)

    @property
    def has_aync_op(self):
        return self.pending_op is not None

    def activate_profiler(self, kn: str, vol: float, backend: str = "nccl", async_op: bool = False):
        self.pending_metadata = (kn, _get_code_location(self.depth), vol, backend, async_op, time.time())
        if self.use_profile:
            self.profiler = profile(enabled=True, use_cuda=True, use_cpu=True, use_kineto=True)
            self.profiler.__enter__()
    
    def activate_p2p_profiler(self):
        if self.use_profile:
            self.profiler = profile(enabled=True, use_cuda=True, use_cpu=True, use_kineto=True)
            self.profiler.__enter__()


    def close_profiler(self, group=None):
        kernel_name, code_location, vol, backend, async_op, start_time = self.pending_metadata
        curr_event = CommEvent()
        if dist.get_world_size(group) > 1:
            if self.use_profile and self.profiler.enabled:
                assert self.profiler is not None, "There is no running dist op"
                self.profiler.__exit__(None, None, None)
                print(self.profiler.function_events, flush=True)
                sync_time = 0
                for event in self.profiler.function_events:
                    event: FunctionEvent
                    if event.name == "cudaDeviceSynchronize":
                        sync_time = event.self_cpu_time_total
                        continue
                    elif event.name == "Memcpy HtoD":
                        curr_event.h2d_time += event.cuda_time_total
                        curr_event.h2d_count += 1
                    elif event.name == "Memcpy DtoH":
                        curr_event.d2h_count += 1
                        curr_event.d2h_time += event.cuda_time_total
                    elif event.name == "aten::copy_":
                        if len(event.input_shapes) == 0 or len(
                                event.input_shapes[0]) == 0 or event.cuda_time_total == 0 or len(event.stack) == 0:
                            continue
                        curr_event.copy_cuda_time = event.cuda_time_total

                    if kernel_name in event.name:
                        curr_event.self_count = 1
                        curr_event.self_comm_vol = vol
                        curr_event.self_cuda_time = event.self_cuda_time_total

                if curr_event.self_count == 0:
                    curr_event.self_count = 1
                    curr_event.self_comm_vol = vol
                    curr_event.self_cuda_time = sync_time
                else:
                    pass
                    # print("DO all reduce", flush=True)
                    # buffer = torch.tensor([curr_event.self_cuda_time], device=get_current_device())
                    # torch_all_reduce(buffer, op=ReduceOp.MIN, group=group)
                    # curr_event.self_cuda_time = buffer.item()

                if curr_event.self_count != 1:
                    if dist.get_rank() == 0:
                        print("dist op num is not equal with 1", flush=True)
                        print(self.profiler.function_events, flush=True)
                    self.logger.error("The number of communication primitives != 1.")
            else:
                torch.cuda.synchronize(get_current_device())
                total_cuda_time = time.time() - start_time
                curr_event.self_count = 1
                curr_event.self_comm_vol = vol
                curr_event.self_cuda_time = total_cuda_time

            # print("kernel_name:{}, code_location:{}, vol:{}, backend:{}, async_op:{}".format(kernel_name, code_location, vol, backend, async_op), flush=True)
            self.add_total(curr_event)
            self.add_record(code_location, curr_event)

        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None
    
    def close_p2p_profiler(self, group=None):
        send_vol, recv_vol, prefix, start_time, op_list = self.pending_p2p
        assert self.aysnc_p2p == 0, "when close profiler must no pending aync p2p"
        if dist.get_world_size(group) > 1:
            if self.use_profile and self.profiler.enabled:
                assert self.profiler is not None, "There is no running dist op"
                assert self.pending_metadata is None, "Collective communication operations should not occur within the scope of p2p operations."
                
                # print("close_profiler, kn: p2p", flush=True)
                self.profiler.__exit__(None, None, None)
                send_kn_name, recv_kn_Name = "c10d::send", "c10d::recv"
                # print("Rank:{}, send_op_count:{}, recv_op_count:{}".format(dist.get_rank(), self.send_op_count, self.recv_op_count), flush=True)
                # print(self.profiler.function_events, flush=True)
                send_event, recv_event = CommEvent(), CommEvent()
                for event in self.profiler.function_events:
                    event: FunctionEvent
                    if send_kn_name in event.name:
                        send_event.self_count += 1
                        send_event.self_comm_vol = send_vol # we already add comm vol
                        # TODO(wgt): send/recv is performed concurrently, and the direct accumulation of 'self_cuda_time_total' here may not be accurate.
                        send_event.self_cuda_time += event.self_cuda_time_total 
                    elif recv_kn_Name in event.name:
                        recv_event.self_count += 1
                        recv_event.self_comm_vol = recv_vol # we already add comm vol
                        # TODO(wgt): send/recv is performed concurrently, and the direct accumulation of 'self_cuda_time_total' here may not be accurate.
                        recv_event.self_cuda_time += event.self_cuda_time_total

                assert send_event.self_count == self.send_op_count, "The expected number of send operations does not match."
                assert recv_event.self_count == self.recv_op_count, "The expected number of send operations does not match."

                self.add_total(send_event)
                self.add_total(recv_event)
                self.add_record(prefix + "/c10d::send", send_event)
                self.add_record(prefix + "/c10d::recv", recv_event)
            else:
                event = CommEvent()
                torch.cuda.synchronize(get_current_device())
                total_cuda_time = time.time() - start_time
                event.self_count = self.send_op_count + self.recv_op_count
                event.self_comm_vol = send_vol or recv_vol  # we only count unidirectional bw.
                event.self_cuda_time = total_cuda_time
                self.add_total(event)
                self.add_record(prefix + "/c10d::send/recv", event)

        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None

        self.pending_p2p = None
        self.send_op_count = 0
        self.recv_op_count = 0
        self.aysnc_p2p = 0
        
    def wait_async_op(self, add_index):
        if self.pending_op is not None:
            op = self.pending_op
            op.wait()
            self.close_profiler()
        
        if self.pending_p2p:
            send_vol, recv_vol, prefix, start_time, op_list = self.pending_p2p
            if add_index != 0:
                op_list[add_index-1].wait()
                self.aysnc_p2p -= 1
                if self.aysnc_p2p == 0:
                    self.close_p2p_profiler()


class CommHandler(object):
    """Communication handler. A dummy handler to wait aync operations.
    """

    def __init__(self, profiler: CommProfiler, add_index=0):
        super().__init__()
        self.prof = profiler
        self.add_index = add_index  # only p2p, and we count start from index 1.

    def wait(self):
        self.prof.wait_async_op(self.add_index)


def async_check(profiler: CommProfiler, p2p : bool= False):
    if not p2p:
        # We suppose that current p2p communication group is over when we encounter a new collective communication op.
        if profiler.send_op_count != 0 or profiler.recv_op_count != 0:
            profiler.close_p2p_profiler()

    if profiler.pending_op is not None:
        profiler.warn_flag = True
        profiler.wait_async_op(0)


def all_reduce(tensor: torch.Tensor,
               op: ReduceOp = ReduceOp.SUM,
               group=None,
               async_op: bool = False,
               profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = 2 * (comm_size - 1) / comm_size
    comm_vol = correction * tensor.element_size() * tensor.numel()
    profiler.activate_profiler("ncclKernel_AllReduce_", comm_vol)
    profiler.pending_op = torch_all_reduce(tensor, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def reduce_scatter(output: torch.Tensor,
                   input_list: List[torch.Tensor],
                   op: ReduceOp = ReduceOp.SUM,
                   group=None,
                   async_op: bool = False,
                   profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for tensor in input_list:
        comm_vol += tensor.element_size() * tensor.numel()
    comm_vol *= correction
    profiler.activate_profiler("ncclKernel_ReduceScatter_", comm_vol)
    profiler.pending_op = torch_reduce_scatter(output, input_list, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def all_gather(tensor_list: List[torch.Tensor],
               tensor: torch.Tensor,
               group=None,
               async_op: bool = False,
               profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for ten in tensor_list:
        comm_vol += ten.element_size() * ten.numel()
    comm_vol *= correction
    profiler.activate_profiler("ncclKernel_AllGather_", comm_vol)
    profiler.pending_op = torch_all_gather(tensor_list, tensor, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)

def all_gather_into_tensor(output: torch.Tensor,
               input_tensor: torch.Tensor,
               group=None,
               async_op: bool = False,
               profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = output.element_size() * output.numel()
    comm_vol *= correction
    profiler.activate_profiler("ncclKernel_AllGather_", comm_vol)
    profiler.pending_op = torch_all_gather_into_tensor(output, input_tensor, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)
    

def broadcast(tensor: torch.Tensor,
              src: int,
              group=None,
              async_op: bool = False,
              profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    backend = get_backend(group)
    # TODO(wgt): add more check.
    if get_backend(group) != "nccl":
        profiler.logger.warrning("Can't profile non-nccl backend communication op.",)
    profiler.activate_profiler("ncclKernel_Broadcast_", comm_vol, backend, async_op)
    profiler.pending_op = torch_broadcast(tensor, src, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def reduce(tensor: torch.Tensor,
           dst: int,
           op: ReduceOp = ReduceOp.SUM,
           group=None,
           async_op: bool = False,
           profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    profiler.activate_profiler("ncclKernel_Reduce_", comm_vol)
    profiler.pending_op = torch_reduce(tensor, dst, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)

def send(tensor: torch.Tensor, dst: int, group = None, tag: int = 0, profiler: CommProfiler = None) -> Optional[CommHandler]:
    """
    If there are multiple send/recv pairs, we will merge the profiling of these p2p operations to 
    avoid the risk of deadlock caused by cudadevicesynchronize.
    """
    async_check(profiler, True)
    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    profiler.find_frist_init_p2p_op(comm_vol, _get_code_location(3), True, None)

    torch_send(tensor, dst, group, tag)

def recv(tensor: torch.Tensor, dst: int, group = None, tag: int = 0, profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler, True)
    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    profiler.find_frist_init_p2p_op(comm_vol, _get_code_location(3), False, None)

    torch_recv(tensor, dst, group, tag)


def isend(tensor: torch.Tensor, dst: int, group= None, tag: int = 0,  profiler: CommProfiler = None) -> Optional[CommHandler]:
    # async_check(profiler, True)
    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    handler = torch_isend(tensor, dst, group, tag)
    profiler.find_frist_init_p2p_op(comm_vol, _get_code_location(3), True, handler)
    # return CommHandler(profiler, profiler.get_p2p_index())
    return handler


def irecv(tensor: torch.Tensor, src: Optional[int] = None, group = None, tag: int = 0,  profiler: CommProfiler = None) -> Optional[CommHandler]:
    # async_check(profiler, True)
    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    handler = torch_irecv(tensor, src, group, tag)
    profiler.find_frist_init_p2p_op(comm_vol, _get_code_location(3), False, handler)
    # return CommHandler(profiler, profiler.get_p2p_index())
    return handler

def batch_isend_irecv(p2p_op_list, profiler: CommProfiler = None):
    reqs_ = []
    reqs = torch_batch_isend_irecv(p2p_op_list)
    for i, req in enumerate(reqs):
        op = p2p_op_list[i]
        comm_vol = 1.0 * op.tensor.element_size() * op.tensor.numel()
        is_send = True if op is dist.isend else False
        profiler.find_frist_init_p2p_op(comm_vol, _get_code_location(3), is_send, req)
        reqs_.append(CommHandler(profiler, profiler.get_p2p_index()))
    return reqs_

    