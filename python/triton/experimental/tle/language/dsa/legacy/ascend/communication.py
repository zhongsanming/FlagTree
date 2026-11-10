# Copyright 2026- Xcoresigma Technology Co., Ltd

import os
import time
import torch
import torch.distributed as dist
import shmem as ash

_g_pe = None
_g_world_size = None
_g_mem_count = 0


def _get_ash_ip_port():
    # Prefer the explicit ASH_* settings, then the torchrun rendezvous port so
    # repeated/multi-run jobs do not collide on the hard-coded 8666 default.
    ash_port = os.environ.get("ASH_MASTER_PORT") or os.environ.get("MASTER_PORT") or "8666"
    ash_addr = os.environ.get("ASH_MASTER_ADDR") or os.environ.get("MASTER_ADDR") or "127.0.0.1"
    return f"tcp://{ash_addr}:{ash_port}"


def _default_ash_size():
    # 1 GiB was enough to make some older SHMEM builds fail at init on 950.
    # The tutorials only need a small symmetric heap; allow an override.
    raw = os.environ.get("ASH_LOCAL_MEM_SIZE", str(256 * 1024 * 1024))
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"ASH_LOCAL_MEM_SIZE must be an integer byte count, got {raw!r}") from None


def init_communicator(ip_port=None, ash_size=None, engine_type=None):
    global _g_pe, _g_world_size, _g_mem_count
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "tle.init_communicator() must be launched with torchrun "
            "(for example: torchrun --nproc-per-node=2 <script>); LOCAL_RANK is unset")
    local_rank = int(os.environ["LOCAL_RANK"])
    # A preceding torchrun/shmem process may still be releasing the device.
    # Retry briefly instead of failing the whole job with Resource_Busy.
    for attempt in range(3):
        try:
            torch.npu.set_device(local_rank)
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(2)
    dist.init_process_group(backend="hccl", rank=local_rank)
    _g_pe = dist.get_rank()
    _g_world_size = dist.get_world_size()
    dist.barrier()

    ash.set_conf_store_tls(False, "")
    attributes = ash.InitAttr()
    attributes.my_rank = _g_pe
    attributes.n_ranks = _g_world_size
    attributes.local_mem_size = ash_size if ash_size is not None else _default_ash_size()
    attributes.ip_port = ip_port if ip_port is not None else _get_ash_ip_port()
    # ROCE/RDMA is the inter-node (multi-machine) protocol; MTE is intra-node only.
    if engine_type is None:
        engine_type = ash.OpEngineType.MTE
    attributes.option_attr.data_op_engine_type = engine_type
    try:
        ret = ash.aclshmem_init(attributes)
    except Exception:
        dist.destroy_process_group()
        raise
    if ret != 0:
        dist.destroy_process_group()
        raise ValueError(
            f"[ERROR] aclshmem_init failed (ret={ret}, rank={_g_pe}, world_size={_g_world_size}, "
            f"port={attributes.ip_port}, local_mem_size={attributes.local_mem_size})")
    _g_mem_count = 0


def create_dist_tensor(buf_tensor):
    """Allocate share memory for distributed communication.

    Each successful allocation increments the global `_g_mem_count` by 1.
    The communicator is finalized only when `cleanup_communicator` brings the
    count back to 0.
    """
    global _g_mem_count
    if isinstance(buf_tensor, torch.Tensor):
        peer_mem = ash.aclshmem_create_tensor([buf_tensor.numel()], dtype=buf_tensor.dtype, device_id=_g_pe)
        _g_mem_count += 1
        return peer_mem
    else:
        raise ValueError("[ERROR] buf_tensor must be a tensor")


def cleanup_communicator(peer_mem):
    """Free one distributed tensor.

    Decrements the global `_g_mem_count` by 1. Only when the count reaches 0
    (i.e. all tensors created by `create_dist_tensor` have been freed) is
    `aclshmem_finalize()` called.
    """
    global _g_mem_count
    torch.npu.synchronize()
    dist.barrier()
    ash.aclshmem_free_tensor(peer_mem)
    _g_mem_count -= 1
    if _g_mem_count < 0:
        raise ValueError("[ERROR] cleanup_communicator called more times than create_dist_tensor")
    if _g_mem_count == 0:
        ash.aclshmem_finalize()
