from functools import partial
from itertools import groupby

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.utils.cuda import get_current_device

from colossalai.utils import is_using_pp

from torch._six import inf
from colossalai.utils.common import _calc_l2_norm, _calc_lp, _get_tensor_norm, _move_norm_to_cuda
from colossalai.constants import IS_TENSOR_PARALLEL
from colossalai.tensor import ColoTensor
from colossalai.tensor.distspec import DistPlacementPattern

from ._utils import (
    flatten,
    get_grad_accumulate_object,
    has_inf_or_nan,
    reduce_tensor,
    release_param_grad,
    split_half_float_double,
    sync_param,
)
from .bookkeeping import BucketStore, GradientStore, ParameterStore, TensorBucket
from colossalai.utils.megatron_timers import timer


class LowLevelZeroOptimizer(ColossalaiOptimizer):
    """Optimizer used for ZeRO-1 and ZeRO-2.
    """

    def __init__(
            self,
            optimizer: Optimizer,

    # grad scaler config
            initial_scale=2**16,
            min_scale=1,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=2000,
            hysteresis=2,
            max_scale: int = 2**24,

    # grad clipping
            clip_grad_norm=0.0,
            verbose=False,

    # communication
            reduce_bucket_size=1024 * 1024,
            communication_dtype=None,
            overlap_communication=False,

    # stage 2
            partition_grad=False,
            dp_parallel_mode=ParallelMode.DATA,
            mp_parallel_mode=ParallelMode.MODEL,

    # cpu offload
            cpu_offload=False,

    # forced dtype
            forced_dtype=None):

        super(LowLevelZeroOptimizer, self).__init__(optim=optimizer)
        if is_using_pp() and overlap_communication:
            raise RuntimeError("The pipeline parallelism is not compatible with overlap_communication, please set overlap_communication=False if you want to use the pipeline parallelism.")
        if is_using_pp() and partition_grad:
            raise RuntimeError("The pipeline parallelism is not compatible with Zero2, please set partition_grad=False if you want to use the pipeline parallelism.")

        self._dtype = self.optim.param_groups[0]['params'][0].dtype
        self._logger = get_dist_logger()
        self._verbose = verbose

        # stage 2
        self._partition_grads = partition_grad

        # cpu_offload
        self._cpu_offload = cpu_offload

        # get process groups
        self._dp_parallel_mode = dp_parallel_mode
        self._mp_parallel_mode = mp_parallel_mode
        self._local_rank = gpc.get_local_rank(dp_parallel_mode)
        self._world_size = gpc.get_world_size(dp_parallel_mode)

        self._dp_group = gpc.get_group(dp_parallel_mode)
        if gpc.is_initialized(mp_parallel_mode) and gpc.get_world_size(mp_parallel_mode) > 1:
            self._mp_group = gpc.get_group(mp_parallel_mode)
        else:
            self._mp_group = None

        # fp16 and fp32 params for mixed precision training
        self._fp16_param_groups = dict()
        self._fp32_flat_param_groups_of_current_rank = dict()

        # communication params
        self._overlap_communication = overlap_communication
        self._reduce_bucket_size = reduce_bucket_size
        self._communication_dtype = communication_dtype

        # gradient scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale,
                                             verbose=verbose)
        self._found_overflow = torch.FloatTensor([0]).to(get_current_device())

        # gradient clipping
        self._clip_grad_norm = clip_grad_norm

        if forced_dtype:
            for group in self.optim.param_groups:
                group_params = group['params']
                for param in group_params:
                    param.data = param.data.to(forced_dtype)
            self._dtype = forced_dtype

        # check argument conflict
        self._sanity_checks()

        # ParameterStore will manage the tensor buffers used for zero
        # it will not manage the tensors used by mixed precision training
        self._param_store = ParameterStore(self._dp_parallel_mode)
        self._grad_store = GradientStore(self._dp_parallel_mode)
        self._bucket_store = BucketStore(self._dp_parallel_mode)

        # iterate over the param group in the optimizer
        # partition these param groups for data parallel training
        # and add buffers to parameter store for future access
        for group_id, param_group in enumerate(self.optim.param_groups):
            group_params = param_group['params']

            # add the fp16 params to fp16_param_groups for bookkeeping
            self._fp16_param_groups[group_id] = group_params

            # assign parameters to ranks
            # the params in the list are sorted
            params_per_rank = self._partition_param_list(group_params)

            # store the mapping between param to rank
            # each param should belong to only one rank
            for rank, params in enumerate(params_per_rank):
                self._param_store.add_fp16_param_list_by_rank_group(rank, group_id, params)
                for param in params:
                    self._param_store.set_param_to_rank(param, rank)

            # move to cpu to make room to create the flat tensor
            # move_tensor(params, device='cpu')
            for param in group_params:
                param.data = param.data.cpu()

            # flatten the reordered tensors
            for rank in range(self._world_size):
                tensor_list = self._param_store.get_fp16_params_by_rank_group(rank, group_id)
                with torch.no_grad():
                    flat_tensor = flatten(tensor_list)
                flat_tensor = flat_tensor.data.cuda()
                self._param_store.add_flat_fp16_param_by_rank_group(rank, group_id, flat_tensor)

            # sync parameters
            for rank in range(self._world_size):
                flat_tensor = self._param_store.get_flat_fp16_param_by_rank_group(rank, group_id)
                tensor_list = self._param_store.get_fp16_params_by_rank_group(rank, group_id)
                sync_param(flat_tensor=flat_tensor, tensor_list=tensor_list)  # 似乎是通过这一步，flat的param和之前的就对接起来了

            # create a copy of fp32 weights of the parameters for which this rank is responsible
            fp16_flat_current_rank = self._param_store.get_flat_fp16_param_by_rank_group(self._local_rank, group_id)
            fp32_flat_current_rank = fp16_flat_current_rank.float()
            device = 'cpu' if self._cpu_offload else get_current_device()
            fp32_flat_current_rank = fp32_flat_current_rank.to(device)
            fp32_flat_current_rank.requires_grad = True
            self._fp32_flat_param_groups_of_current_rank[group_id] = fp32_flat_current_rank

            # need to replace the params in the `params` field in the optimizer
            # so that when the optimizer calls step(), it only updates the tensors
            # managed by this data parallel rank
            param_group['params'] = [fp32_flat_current_rank]

            # set reduction state
            for param in self._fp16_param_groups[group_id]:
                self._param_store.set_param_reduction_state(param, False)
        assert len(self._fp16_param_groups) != 0
        # intialize communication stream for
        # communication-compuation overlapping
        if self._overlap_communication:
            self._comm_stream = torch.cuda.Stream()

        # reduction hook is only used if overlapping communication
        # or stage 2 is used
        # if it is stage 1 without overlapping, no hook will be attached
        if self._overlap_communication or self._partition_grads:
            self._attach_reduction_hook()

    @property
    def dtype(self):
        return self._dtype

    @property
    def loss_scale(self):
        return self.grad_scaler.scale

    @property
    def num_param_groups(self):
        return len(self._fp16_param_groups)

    def _partition_param_list(self, param_list):
        params_per_rank = [[] for _ in range(self._world_size)]
        numel_per_rank = [0 for _ in range(self._world_size)]

        # partititon the parameters in a greedy fashion
        sorted_params = sorted(param_list, key=lambda x: x.numel(), reverse=True)
        for param in sorted_params:
            # allocate this parameter to the rank with
            # the smallest numel for load balancing purpose
            rank_to_go = numel_per_rank.index(min(numel_per_rank))
            params_per_rank[rank_to_go].append(param)
            numel_per_rank[rank_to_go] += param.numel()

        if self._verbose:
            self._logger.info(f'Number of elements on ranks: {numel_per_rank}, rank:{gpc.get_global_rank()}',
                              ranks=[0],
                              parallel_mode=self._dp_parallel_mode)
        return params_per_rank

    def _sanity_checks(self):
        assert torch.cuda.is_available(), 'CUDA is required'
        for param_group in self.optim.param_groups:
            group_params = param_group['params']
            for param in group_params:
                assert param.dtype == self._dtype, \
                    f"Parameters are expected to have the same dtype `{self._dtype}`, but got `{param.dtype}`"

    ###########################################################
    # Backward Reduction Hook
    ###########################################################

    def _attach_reduction_hook(self):
        # we iterate over the fp16 params
        # on each param, we register a hook to its AccumulateGrad object
        for group_id in range(self.num_param_groups):
            param_group = self._fp16_param_groups[group_id]
            for param in param_group:
                if param.requires_grad:
                    # determines the reduction destionation rank
                    # this is only valid for stage 2
                    # dst_rank = None means using all-reduce
                    # else using reduce
                    if self._partition_grads:
                        reduce_rank = self._param_store.get_param_rank(param)
                    else:
                        reduce_rank = None

                    def _define_and_attach(param, reduce_rank):
                        # get the AccumulateGrad object of the param itself
                        accum_grad_obj = get_grad_accumulate_object(param)
                        self._grad_store.add_accumulate_grad_object(accum_grad_obj)

                        reduction_func = partial(self._reduce_and_remove_grads_by_bucket,
                                                 param=param,
                                                 reduce_rank=reduce_rank)

                        # define hook
                        # NOT IMPORTANT BUT GOOD TO KNOW:
                        # args here is not grad, but allow_unreacable and accumulate_grad
                        def reduce_grad_hook(*args):
                            reduction_func()

                        accum_grad_obj.register_hook(reduce_grad_hook)

                    _define_and_attach(param, reduce_rank)

    def _reduce_and_remove_grads_by_bucket(self, param, reduce_rank=None):
        param_size = param.numel()

        # check if the bucket is full
        # if full, will reduce the grads already in the bucket
        # after reduction, the bucket will be empty
        if self._bucket_store.num_elements_in_bucket(reduce_rank) + param_size > self._reduce_bucket_size:
            self._reduce_grads_in_bucket(reduce_rank)

        # the param must not be reduced to ensure correctness
        is_param_reduced = self._param_store.is_param_reduced(param)
        if is_param_reduced:
            msg = f'Parameter of size ({param.size()}) has already been reduced, ' \
                  + 'duplicate reduction will lead to arithmetic incorrectness'
            raise RuntimeError(msg)

        # the param must have grad for reduction
        assert param.grad is not None, f'Parameter of size ({param.size()}) has None grad, cannot be reduced'

        self._bucket_store.add_num_elements_in_bucket(param_size, reduce_rank)
        self._bucket_store.add_grad(param.grad, reduce_rank)
        self._bucket_store.add_param(param, reduce_rank)

    def _reduce_grads_in_bucket(self, reduce_rank=None):
        # reduce grads
        self._reduce_grads_by_rank(reduce_rank=reduce_rank,
                                   grads=self._bucket_store.get_grad(reduce_rank=reduce_rank),
                                   bucket_size=self._bucket_store.num_elements_in_bucket(reduce_rank))

        # use communication stream if overlapping
        # communication with computation
        if self._overlap_communication:
            stream = self._comm_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            params_in_bucket = self._bucket_store.get_param(reduce_rank=reduce_rank)

            for param in params_in_bucket:
                # the is_param_reduced flag should be False showing that
                # this param is not reduced before calling self._reduce_grads_by_rank
                is_param_reduced = self._param_store.is_param_reduced(param)

                if is_param_reduced:
                    msg = f'Parameter of size ({param.size()}) has been reduced, ' + \
                          'duplicate reduction will lead to arithmetic incorrectness'
                    raise RuntimeError(msg)

                # update the flag
                self._param_store.set_param_reduction_state(param, True)

                # if partition grads = True
                # we do not keep the gradient after reduction
                if self._partition_grads and not self._param_store.belongs_to_current_rank(param):
                    if self._overlap_communication:
                        # we need to keep this gradient for now as reduction may
                        # be completed yet since it is using a different cuda stream
                        self._param_store.add_previous_reduced_param(param)
                    else:
                        param.grad = None

        self._bucket_store.reset_by_rank(reduce_rank)

    def _reduce_grads_by_rank(self, reduce_rank, grads, bucket_size):
        grad_buckets_by_dtype = split_half_float_double(grads)

        for tensor_list in grad_buckets_by_dtype:
            self._reduce_no_retain(tensor_list=tensor_list, bucket_size=bucket_size, reduce_rank=reduce_rank)

    ##############################
    # Reduction Utility Function #
    ##############################
    def _reduce_no_retain(self, tensor_list, bucket_size, reduce_rank):
        param_bucket = TensorBucket(size=bucket_size)

        for tensor in tensor_list:
            param_bucket.add_to_bucket(tensor, allow_oversize=True)

            if param_bucket.is_full_or_oversized():
                self._reduce_and_copy(bucket=param_bucket, reduce_rank=reduce_rank)
                param_bucket.empty()

        if not param_bucket.is_empty():
            self._reduce_and_copy(bucket=param_bucket, reduce_rank=reduce_rank)

    def _reduce_and_copy(self, bucket: TensorBucket, reduce_rank):
        if self._overlap_communication:
            torch.cuda.synchronize()
            self._param_store.clear_grads_of_previous_reduced_params()
            stream = self._comm_stream
        else:
            stream = torch.cuda.current_stream()

        with torch.cuda.stream(stream):
            flat = bucket.flatten()
            reduced_flat = reduce_tensor(tensor=flat,
                                         dtype=self._communication_dtype,
                                         dst_rank=reduce_rank,
                                         parallel_mode=self._dp_parallel_mode)

            # update the reduced tensor
            if reduce_rank is None or reduce_rank == self._local_rank:
                bucket.unflatten_and_copy(reduced_flat)

    ################################
    # torch.optim.Optimizer methods
    ################################

    def backward(self, loss, retain_graph=False):
        loss = self.loss_scale * loss
        loss.backward(retain_graph=retain_graph)

        # finish gradient reduction
        if not is_using_pp():
            self._reduce_grad()
    
    def _reduce_grad(self):
        if not self._partition_grads:
            self._reduce_grad_stage1()
        else:
            # TODO: support async comm in reduce
            self._reduce_grad_stage2()

        # clear reduced grads
        if self._overlap_communication:
            torch.cuda.synchronize()
            self._param_store.clear_grads_of_previous_reduced_params()

    def zero_grad(self, set_to_none=True):
        """
        Set parameter gradients to zero. If set_to_none = True, gradient
        will be set to None to save memory.

        :param set_to_none: Whether set the gradient to None. Default value is True.
        :type set_to_none: bool
        """
        for group_id, param_group in self._fp16_param_groups.items():
            for param in param_group:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.detach()
                        param.grad.zero_()

    ####################
    # Update Parameter #
    ####################

    def step(self, closure=None):
        if is_using_pp():
            timer('sync_grad').start()
            self._reduce_grad()
            self.sync_grad()
            timer('sync_grad').stop()
        return self._step(closure=closure)


    def _step(self, closure=None):
        assert closure is None, 'closure is not supported by step()'

        # check for overflow
        found_inf = self._check_overflow()
        loss_scale = float(self.loss_scale.item())  # backup, later it will be used for computing
        self.grad_scaler.update(found_inf)

        # update loss scale if overflow occurs
        if found_inf:
            self._grad_store._averaged_gradients = dict()
            self.zero_grad()
            return False, None

        # copy the grad of fp16 param to fp32 param
        single_grad_partition_groups = []
        norm_groups = []
        global_norm = None
        timer('cal_norm').start()
        for group_id in range(self.num_param_groups):
            # compute norm
            gradients = self._grad_store.get_averaged_gradients_by_group(group_id)
            if self._clip_grad_norm > 0:
                # this norm is before scaling, will be very large
                norm_group = compute_norm(gradients=gradients,
                                        parameters=self._param_store.get_fp16_params_by_rank_group(group_id=group_id,
                                                                                                rank=self._local_rank))  # before scale norm
                norm_groups.append(norm_group)

            # create flat gradient for the flat fp32 params
            fp16_avg_grads = gradients
            flat_fp16_avg_grads = flatten(fp16_avg_grads)

            dtype = self._fp32_flat_param_groups_of_current_rank[group_id].dtype
            flat_fp32_avg_grads = flat_fp16_avg_grads.to(dtype)

            param_shape = self._fp32_flat_param_groups_of_current_rank[group_id].shape
            assert param_shape == flat_fp32_avg_grads.shape, \
                f'fp32 param and grad have different shape {param_shape} vs {flat_fp32_avg_grads.shape}'

            single_grad_partition_groups.append(flat_fp32_avg_grads)
            device = self._fp32_flat_param_groups_of_current_rank[group_id].device
            self._fp32_flat_param_groups_of_current_rank[group_id].grad = flat_fp32_avg_grads.to(device)
            self._grad_store._averaged_gradients[group_id] = []
            self._grad_store._averaged_gradients[group_id] = []

        # unscale and clip grads
        # get the global norm
        if self._clip_grad_norm>0:
            global_norm = sum(norm_groups)**0.5
        self._unscale_and_clip_grads(single_grad_partition_groups, global_norm, loss_scale)
        timer('cal_norm').stop()
        # update the parameters
        timer('step').start()
        self.optim.step()
        # release the fp32 grad
        release_param_grad(self._fp32_flat_param_groups_of_current_rank.values())

        # update fp16 partition updated by the current rank
        for group_id in range(len(self._fp16_param_groups)):
            fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=self._local_rank, group_id=group_id)
            fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
            fp16_param.data.copy_(fp32_param)

        # broadcast the updated model weights
        handles = []
        for group_id in range(self.num_param_groups):
            for rank in range(self._world_size):
                fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=rank, group_id=group_id)
                rank = gpc.get_ranks_in_group(ParallelMode.DATA)[rank]  # need to convert to the global rank
                handle = dist.broadcast(fp16_param, src=rank, group=self._dp_group, async_op=True)
                handles.append(handle)

        for handle in handles:
            handle.wait()
        timer('step').stop()
        # 这里可能不需要 update gradients , 是由于在初始化中使用 sync_params 函数，所以保持了同步
        return True, global_norm/loss_scale

    ##################
    # FP16 Utilities #
    ##################

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group_id in range(len(self._fp16_param_groups)):
            for avg_grad in self._grad_store.get_averaged_gradients_by_group(group_id):
                if avg_grad is not None and has_inf_or_nan(avg_grad):
                    self._found_overflow.fill_(1.0)
                    break

        # all-reduce across dp group
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self._dp_group)

        # all-reduce over model parallel group
        if self._mp_group:
            dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self._mp_group)

        if self._found_overflow.item() > 0:
            return True
        else:
            return False

    def _unscale_and_clip_grads(self, grad_groups_flat, total_norm, loss_scale):
        # compute combined scale factor for this group
        combined_scale = loss_scale

        if self._clip_grad_norm > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / loss_scale) + 1e-6) / self._clip_grad_norm
            if clip > 1.:
                combined_scale = clip * loss_scale

        for grad in grad_groups_flat:
            grad.data.mul_(1. / combined_scale)

    ############################
    # Gradient Synchronization #
    ############################

    def sync_grad(self):
        # update param already reduced flag
        reduction_states = self._param_store.get_param_reduction_states()
        for tensor, state in reduction_states.items():
            reduction_states[tensor] = False

        # accumulate gradient
        avg_gradients = self._grad_store._averaged_gradients
        for group_id in range(self.num_param_groups):
            param_group = self._param_store.get_fp16_params_by_rank_group(self._local_rank, group_id)

            if group_id not in avg_gradients:
                avg_gradients[group_id] = []

            param_idx = 0
            for param in param_group:
                if param.grad is not None:
                    if len(avg_gradients[group_id]) == param_idx:
                        avg_gradients[group_id].append(param.grad)
                    else:
                        avg_gradients[group_id][param_idx].add_(param.grad)
                    param_idx += 1

        # the gradients needed are stored in the avg_gradients buffer
        # thus, can clear this
        self.zero_grad()

    def _reduce_grad_stage1(self):
        # if not overlapping communication (no reduction hook is attached)
        # we need to manually reduce these gradients
        if not self._overlap_communication:
            for group_id in range(len(self._fp16_param_groups)):
                param_group = self._fp16_param_groups[group_id]
                for param in param_group:
                    if param.grad is not None:
                        self._reduce_and_remove_grads_by_bucket(param)

        # we need to reduce the gradients
        # left in the communication bucket
        self._reduce_grads_in_bucket()

    def _reduce_grad_stage2(self):
        # when partition_grads is True, reduction hooks
        # are attached in the __init__ function, so we
        # only need to reduce the gradients
        # left in the communication bucket
        for reduce_rank in range(self._world_size):
            self._reduce_grads_in_bucket(reduce_rank)

    def clip_grad_norm(self, model, max_norm):
        # will conduct in the step()
        pass

    # TODO 需要加入state_dict方法
    def state_dict(self):
        states = {}
        grad_scaler = self.grad_scaler.state_dict()
        # TODO 需要考虑没有 grad_scaler 的情况
        states['grad_scaler'] = grad_scaler

        # 传入的 optimizer 的 state , todo 如果这个optimizer还没跑过就state dict的话，可能会导致其中的一些东西不存在，可能会在之后报错？
        optim_states = self.optim.state_dict()  
        states['base_optim_states'] = optim_states

        # 自身管理的 fp32 的权重部分
        flat_fp32_weights = {}
        for group_id, param in self._fp32_flat_param_groups_of_current_rank.items():
            assert param.grad is None
            flat_fp32_weights[group_id] = param
        states['flat_fp32_weights'] = flat_fp32_weights

        # TODO 应该还需要有一些sanity check的内容

        # TODO 需要考虑出现 dp 数量变化的情况

        return states

    def load_state_dict(self, states):
        # TODO 需要考虑出现 dp 数量变化的情况

        # TODO 需要考虑没有 loss_scaler 的情况
        grad_scaler = states['grad_scaler']
        self.grad_scaler.load_state_dict(grad_scaler)

        # load optimizer
        optim_states = states['base_optim_states']
        self.optim.load_state_dict(optim_states)

        # fp32 权重
        flat_fp32_weights = states['flat_fp32_weights']
        assert set(flat_fp32_weights.keys()) == set(self._fp32_flat_param_groups_of_current_rank)
        for group_id, param in flat_fp32_weights.items():
            _param = self._fp32_flat_param_groups_of_current_rank[group_id]
            assert _param.shape == param.shape
            _param.data.copy_(param.data)
        # 需要对model的进行赋值
        for group_id in range(len(self._fp16_param_groups)):
            fp16_param = self._param_store.get_flat_fp16_param_by_rank_group(rank=self._local_rank, group_id=group_id)
            fp32_param = self._fp32_flat_param_groups_of_current_rank[group_id]
            fp16_param.data.copy_(fp32_param)  # 自动也就改变了 model 那边的值了


def is_model_parallel_parameter(p):
    # colossalai.utils.common.is_model_parallel_parameter cannot process the ColoParam 
    return (hasattr(p, IS_TENSOR_PARALLEL) and getattr(p, IS_TENSOR_PARALLEL)) or \
        (isinstance(p, ColoTensor) and p.dist_spec.placement == DistPlacementPattern.SHARD)


def compute_norm(gradients, parameters, norm_type=2):
    """Get the norm 
    Arguments:
        gradients (Iterable[Tensor]): The gradient value
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to 
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        
    Returns:
        Total norm of the parameters, need total_norm**(1/norm) before using.
    """

    enable_cuda_kernels = gradients[0].device.type == 'cuda'
    # Norm parameters.
    norm_type = float(norm_type)

    has_zero_shared_param: bool = False
    for param in parameters:
        if hasattr(param, 'colo_attr') and param.colo_attr.sharded_data_tensor.is_sharded:
            raise RuntimeError("Currently not support Zero3")

    # Parameters can be on CPU or CUDA
    # If parameters are on CPU, disable CUDA kernerls

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = torch.FloatTensor([float(total_norm)]).to(gradients[0].device)
        # Take max across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.MODEL) and gpc.get_world_size(ParallelMode.MODEL) > 1:
            dist.all_reduce(total_norm_cuda,
                            op=dist.ReduceOp.MAX,
                            group=gpc.get_group(ParallelMode.MODEL),
                            async_op=False)
        if has_zero_shared_param:
            dist.all_reduce(total_norm_cuda, 
                                op=dist.ReduceOp.MAX,
                                group=gpc.get_group(ParallelMode.DATA),
                                async_op=False)
        total_norm = total_norm_cuda[0].item()
    else:
        tensor_parallel_grads = []
        for g, p in zip(gradients, parameters):
            # TODO consider the pipeline shared parameter

            if gpc.is_initialized(ParallelMode.PIPELINE) and hasattr(p, 'pipeline_shared_module_pg') and dist.get_rank(p.pipeline_shared_module_pg)==0:  # if shared between different pipe, only count o
                tensor_parallel_grads.append(g.data.float())
                # if is_model_parallel_parameter(p):
                #     tensor_parallel_grads.append(g.data.float())
                # else:
                #     no_tensor_parallel_grads.append(g.data.float())
            elif gpc.is_initialized(ParallelMode.PIPELINE) and hasattr(p, 'pipeline_shared_module_pg') and dist.get_rank(p.pipeline_shared_module_pg)!=0:
                continue
            elif gpc.is_initialized(ParallelMode.TENSOR) and not is_model_parallel_parameter(p) and gpc.get_local_rank(ParallelMode.TENSOR)==0:  # if not used in each chunk, such as layernorm
                # no_tensor_parallel_grads.append(g.data.float())
                tensor_parallel_grads.append(g.data.float())
            elif is_model_parallel_parameter(p):
                # reductor = (gpc.get_world_size(ParallelMode.TENSOR) / getattr(p, NUM_PARTITIONS))**(1 / norm_type)
                tensor_parallel_grads.append(g.data.float())
            elif gpc.get_local_rank(ParallelMode.TENSOR)!=0:
                continue
            else:
                raise RuntimeError("Should not arrive here")

        if norm_type == 2.0 and enable_cuda_kernels:
            tensor_parallel_norm = _calc_l2_norm(tensor_parallel_grads)**norm_type
            # no_tensor_parallel_norm = _calc_l2_norm(no_tensor_parallel_grads)**norm_type
        else:
            tensor_parallel_norm = _calc_lp(tensor_parallel_grads, norm_type)
            # no_tensor_parallel_norm = _calc_lp(no_tensor_parallel_grads, norm_type)
        
        # If norm is type of float, then we convert them into torch.Tensor.
        tensor_parallel_norm = _get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
        # no_tensor_parallel_norm = _get_tensor_norm(no_tensor_parallel_norm, enable_cuda_kernels)
        # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
        if not enable_cuda_kernels:
            tensor_parallel_norm = _move_norm_to_cuda(tensor_parallel_norm)
            # no_tensor_parallel_norm = _move_norm_to_cuda(no_tensor_parallel_norm)

        # total_norm = tensor_parallel_norm + no_tensor_parallel_norm
        total_norm = tensor_parallel_norm

        # Sum across all model-parallel GPUs.
        if gpc.is_initialized(ParallelMode.MODEL):
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.MODEL))

        # This is because we use zero1, so we need to use this reduction.
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.DATA))

        if torch.is_tensor(total_norm):
            total_norm = total_norm.item()

    # Scale.
    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm

