# Copyright 2026- Xcoresigma Technology Co., Ltd

import os

try:
    import shmem as ash
except ImportError:  # pragma: no cover - depends on the Ascend runtime
    ash = None

_g_pe = None
_g_world_size = None
_g_mem_count = 0


def _require_torch():
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - torch is a runtime dependency
        raise ImportError("tle.dsa.ascend communication requires torch and torch.distributed") from exc
    return torch, dist


def _require_ash():
    if ash is None:
        raise ImportError("tle.dsa.ascend communication requires the 'shmem' package "
                          "(aclshmem) on the Ascend host")
    return ash


def _get_ash_ip_port():
    # default to 8666 if ASH_MASTER_PORT is not set
    ash_port = os.environ.get("ASH_MASTER_PORT", "8666")
    # default to 127.0.0.1 if ASH_MASTER_ADDR is not set
    ash_addr = os.environ.get("ASH_MASTER_ADDR", "127.0.0.1")
    return f"tcp://{ash_addr}:{ash_port}"


def init_communicator(ip_port=None, ash_size=None, engine_type=None):
    global _g_pe, _g_world_size, _g_mem_count
    torch, dist = _require_torch()
    shmem = _require_ash()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=local_rank)
    _g_pe = dist.get_rank()
    _g_world_size = dist.get_world_size()
    dist.barrier()

    shmem.set_conf_store_tls(False, "")
    attributes = shmem.InitAttr()
    attributes.my_rank = _g_pe
    attributes.n_ranks = _g_world_size
    attributes.local_mem_size = ash_size if ash_size is not None else 1024 * 1024 * 1024
    attributes.ip_port = ip_port if ip_port is not None else _get_ash_ip_port()
    # ROCE/RDMA is the inter-node (multi-machine) protocol; MTE is intra-node only.
    if engine_type is None:
        engine_type = shmem.OpEngineType.MTE
    attributes.option_attr.data_op_engine_type = engine_type
    ret = shmem.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")
    _g_mem_count = 0


def create_dist_tensor(buf_tensor):
    """Allocate shared memory for distributed communication."""
    global _g_mem_count
    torch, _ = _require_torch()
    shmem = _require_ash()
    if isinstance(buf_tensor, torch.Tensor):
        peer_mem = shmem.aclshmem_create_tensor([buf_tensor.numel()], dtype=buf_tensor.dtype, device_id=_g_pe)
        _g_mem_count += 1
        return peer_mem
    raise ValueError("[ERROR] buf_tensor must be a tensor")


def cleanup_communicator(peer_mem):
    """Free one distributed tensor."""
    global _g_mem_count
    torch, dist = _require_torch()
    shmem = _require_ash()
    torch.npu.synchronize()
    dist.barrier()
    shmem.aclshmem_free_tensor(peer_mem)
    _g_mem_count -= 1
    if _g_mem_count < 0:
        raise ValueError("[ERROR] cleanup_communicator called more times than create_dist_tensor")
    if _g_mem_count == 0:
        shmem.aclshmem_finalize()
