# Ascend pipe scheduler/helpers - compat layer for 3.6 port.
# Provides pipe_scheduler, run_pipe, SyncSpec-driven event management as used by tutorials.
# Full workspace-pipe construction is handled via tle.dsa.workspace + tle.dsa.ascend.pipe helpers.

from dataclasses import dataclass
from typing import Any, Dict

import triton.language.core as tl
from triton.language.core import _unwrap_if_constexpr, builtin

from .core import PIPE, SyncSpec, sync_block_set, sync_block_wait

_pipe_event_ranges = []


def reset_pipe_event_allocator():
    _pipe_event_ranges.clear()


def _unwrap_constexpr(value):
    return _unwrap_if_constexpr(value)


def _as_int(value, name):
    value = _unwrap_constexpr(value)
    if not isinstance(value, int):
        raise TypeError(f"{name} must be a compile-time integer")
    return value


def _default_sync_specs():
    ready_sync = SyncSpec(sender="cube", receiver="vector", sender_pipe=PIPE.PIPE_FIX, receiver_pipe=PIPE.PIPE_MTE2)
    free_sync = SyncSpec(sender="vector", receiver="cube", sender_pipe=PIPE.PIPE_MTE2, receiver_pipe=PIPE.PIPE_FIX)
    return ready_sync, free_sync


class PipeScheduler:
    __triton_compile_time_value__ = True
    backend = "ascend"

    def __init__(self, ready_sync, free_sync, event_base=None):
        self.ready_sync = ready_sync
        self.free_sync = free_sync
        self.event_base = event_base


@builtin
def pipe_scheduler(ready_sync=None, free_sync=None, event_base=None, _semantic=None):
    ready_sync = _unwrap_constexpr(ready_sync)
    free_sync = _unwrap_constexpr(free_sync)
    event_base = _unwrap_constexpr(event_base)
    if ready_sync is None or free_sync is None:
        default_ready, default_free = _default_sync_specs()
        ready_sync = default_ready if ready_sync is None else ready_sync
        free_sync = default_free if free_sync is None else free_sync
    if ready_sync.sender != free_sync.receiver:
        raise ValueError("ready_sync.sender must equal free_sync.receiver")
    if ready_sync.receiver != free_sync.sender:
        raise ValueError("ready_sync.receiver must equal free_sync.sender")
    if event_base is not None:
        event_base = _as_int(event_base, "event_base")
    return PipeScheduler(ready_sync, free_sync, event_base)


def _allocate_event_base(capacity):
    base = 0
    if _pipe_event_ranges:
        base = max(end for _, end in _pipe_event_ranges)
    _reserve_event_range(base, capacity)
    return base


def _reserve_event_range(event_base, capacity):
    start = event_base
    end = event_base + capacity
    if start < 0 or end > 16:
        raise ValueError("pipe event range invalid/exceeds Ascend event id limit 16")
    for existing_start, existing_end in _pipe_event_ranges:
        if start < existing_end and existing_start < end:
            raise ValueError(f"pipe event range [{start}, {end}) conflicts")
    _pipe_event_ranges.append((start, end))


@dataclass(frozen=True)
class pipe_wait_result:
    __triton_compile_time_value__ = True
    slot: Any
    is_closed: bool = False


class pipe_slot:
    __triton_compile_time_value__ = True

    def __init__(self, fields: Dict[str, Any]):
        self._fields = fields
        for name, value in fields.items():
            setattr(self, name, value)

    def __getitem__(self, key):
        return self._fields[key]


class _pipe_endpoint:
    __triton_compile_time_value__ = True

    def __init__(self, pipe_obj, kind):
        self._pipe = pipe_obj
        self.kind = kind
        self.fields = pipe_obj.fields


class pipe_writer(_pipe_endpoint):
    def __init__(self, pipe_obj):
        super().__init__(pipe_obj, "writer")

    @builtin
    def acquire(self, stage, _semantic=None):
        stage = _unwrap_constexpr(stage)
        return self._pipe.acquire_write(stage, _semantic=_semantic)

    @builtin
    def commit(self, stage, _semantic=None):
        stage = _unwrap_constexpr(stage)
        return self._pipe.commit_write(stage, _semantic=_semantic)


class pipe_reader(_pipe_endpoint):
    def __init__(self, pipe_obj):
        super().__init__(pipe_obj, "reader")

    @builtin
    def wait(self, stage, _semantic=None):
        stage = _unwrap_constexpr(stage)
        slot = self._pipe.wait_read(stage, _semantic=_semantic)
        return pipe_wait_result(slot)

    @builtin
    def release(self, stage, _semantic=None):
        stage = _unwrap_constexpr(stage)
        return self._pipe.release_read(stage, _semantic=_semantic)


class pipe_value:
    __triton_compile_time_value__ = True

    def __init__(self, *, capacity, scope="cta", name=None, fields=None, scheduler=None, _semantic=None):
        self.capacity = _as_int(capacity, "capacity")
        self.scope = scope
        self.name = name
        self.fields = fields or {}
        self.scheduler = scheduler
        self.event_base = None
        if scheduler is not None and getattr(scheduler, 'event_base', None) is not None:
            self.event_base = _as_int(scheduler.event_base, "event_base")
            _reserve_event_range(self.event_base, self.capacity)
        else:
            self.event_base = _allocate_event_base(self.capacity)
        # Initialize: mark buffers as free (vector->cube) for each slot
        self._init_done = False

    def _event_id(self, stage):
        return self.event_base + (stage % self.capacity)

    def _slot(self, stage, _semantic=None):
        f = {}
        for k, v in self.fields.items():
            if hasattr(v, 'slot'):
                f[k] = v.slot(stage, _semantic=_semantic)
            else:
                f[k] = v
        return pipe_slot(f)

    def _init_events(self, _semantic=None):
        if self._init_done:
            return
        sched = self.scheduler
        if sched is None:
            sched = PipeScheduler(*_default_sync_specs())
        for i in range(self.capacity):
            sync_block_set(sched.free_sync.sender, sched.free_sync.receiver, self.event_base + i,
                           sched.free_sync.sender_pipe, sched.free_sync.receiver_pipe, _semantic=_semantic)
        self._init_done = True

    def writer(self):
        return pipe_writer(self)

    def reader(self):
        return pipe_reader(self)

    @builtin
    def acquire_write(self, stage, _semantic=None):
        self._init_events(_semantic=_semantic)
        sched = self.scheduler or PipeScheduler(*_default_sync_specs())
        eid = self._event_id(stage)
        sync_block_wait(sched.free_sync.sender, sched.free_sync.receiver, eid,
                        sched.free_sync.sender_pipe, sched.free_sync.receiver_pipe, _semantic=_semantic)
        return self._slot(stage, _semantic=_semantic)

    @builtin
    def commit_write(self, stage, _semantic=None):
        sched = self.scheduler or PipeScheduler(*_default_sync_specs())
        eid = self._event_id(stage)
        sync_block_set(sched.ready_sync.sender, sched.ready_sync.receiver, eid,
                       sched.ready_sync.sender_pipe, sched.ready_sync.receiver_pipe, _semantic=_semantic)

    @builtin
    def wait_read(self, stage, _semantic=None):
        self._init_events(_semantic=_semantic)
        sched = self.scheduler or PipeScheduler(*_default_sync_specs())
        eid = self._event_id(stage)
        sync_block_wait(sched.ready_sync.sender, sched.ready_sync.receiver, eid,
                        sched.ready_sync.sender_pipe, sched.ready_sync.receiver_pipe, _semantic=_semantic)
        return self._slot(stage, _semantic=_semantic)

    @builtin
    def release_read(self, stage, _semantic=None):
        sched = self.scheduler or PipeScheduler(*_default_sync_specs())
        eid = self._event_id(stage)
        sync_block_set(sched.free_sync.sender, sched.free_sync.receiver, eid,
                       sched.free_sync.sender_pipe, sched.free_sync.receiver_pipe, _semantic=_semantic)


@builtin
def pipe(*, capacity, scope="cta", name=None, fields=None, scheduler=None, readers=None, one_shot=False, _semantic=None, **kwargs):
    """Ascend workspace-pipe constructor (compat for 3.5.x tutorial style)."""
    if fields is None and kwargs:
        fields = kwargs
    return pipe_value(capacity=capacity, scope=scope, name=name, fields=fields, scheduler=scheduler, _semantic=_semantic)


@builtin
def run_pipe(tasks, scheduler=None, _semantic=None):
    """Execute Ascend CV-mix pipe tasks. Each task is (fn, (args...))."""
    # Ascend's dual-core CV model: cube runs producers, vector runs consumers.
    # In the JIT, both are emitted and the backend partitions by core role/syncs.
    # Here we execute sequentially at the Python/semantic level as a compat shim;
    # actual partitioning is handled by compiler passes + sync_block ops.
    for fn, args in tasks:
        fn(*args)
