import pytest
import triton


def get_dsa_backend():
    """Return (backend, device, space) for a supported TLE DSA runtime, or skip."""
    torch = pytest.importorskip("torch")
    import triton.experimental.tle.language as tle
    try:
        target = triton.runtime.driver.active.get_current_target()
        backend = getattr(target, "backend", None)
    except Exception:
        backend = None

    if backend == "txda":
        pytest.importorskip("torch_txda")
        return "txda", "txda", tle.dsa.tsingmicro.SPM

    if backend in ("npu", "ascend"):
        pytest.importorskip("torch_npu")
        return backend, "npu", tle.dsa.ascend.UB

    try:
        pytest.importorskip("torch_npu")
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "ascend", "npu", tle.dsa.ascend.UB
    except pytest.skip.Exception:
        pass
    except Exception:
        pass
    try:
        pytest.importorskip("torch_txda")
        return "txda", "txda", tle.dsa.tsingmicro.SPM
    except pytest.skip.Exception:
        pass
    except Exception:
        pass

    pytest.skip("TLE DSA tests require Ascend (npu/torch_npu) or TsingMicro (txda/torch_txda) backend")
