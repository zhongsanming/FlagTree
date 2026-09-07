"""Minimal runner for mhc_pre_clamp_sinkhorn.

Usage:
    python test_mhc_pre_clamp_sinkhorn.py
    python test_mhc_pre_clamp_sinkhorn.py --device npu --B 1 --S 16 --D 128
    MLIR_ENABLE_DUMP=1 python test_mhc_pre_clamp_sinkhorn.py --dump
    MLIR_ENABLE_DUMP=mhc_pre_clamp_sinkhorn python test_mhc_pre_clamp_sinkhorn.py --dump

This script is intentionally small: it constructs one valid input set, invokes
mhc_pre_clamp_sinkhorn once, and optionally compares against the PyTorch
reference implementation.
"""

from __future__ import annotations

import argparse
import os

import torch

from mhc_pre_clamp_sinkhorn import mhc_pre_clamp_sinkhorn, mhc_pre_clamp_sinkhorn_ref


def _resolve_device(device: str) -> str:
    if device == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def _sync(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _make_inputs(B: int, S: int, N: int, D: int, dtype: torch.dtype, device: str):
    assert N == 4, "mhc_pre_clamp_sinkhorn currently requires N=4"
    hc_mix = N * (N + 2)
    hc_d = N * D

    x = torch.randn((B, S, N, D), dtype=dtype, device=device)
    phi = torch.randn((hc_mix, hc_d), dtype=torch.float32, device=device)
    alpha = torch.randn((3,), dtype=torch.float32, device=device)
    base = torch.randn((hc_mix,), dtype=torch.float32, device=device)
    return x, phi, alpha, base


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal runner for mhc_pre_clamp_sinkhorn")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto/npu/cuda/cpu")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--S", type=int, default=8, help="Sequence length")
    parser.add_argument("--N", type=int, default=4, help="Head multiplier; must be 4")
    parser.add_argument("--D", type=int, default=128, help="Hidden dimension per head")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Input dtype for x")
    parser.add_argument("--iter-times", type=int, default=20, help="Sinkhorn iteration count")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=0.0)
    parser.add_argument("--need-backward", action="store_true", help="Request extra saved tensors")
    parser.add_argument("--check", action="store_true", help="Compare outputs to reference implementation")
    parser.add_argument("--dump", action="store_true", help="Print current MLIR_ENABLE_DUMP setting as a reminder")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    if args.dump:
        print(f"MLIR_ENABLE_DUMP={os.environ.get('MLIR_ENABLE_DUMP', '<unset>')}")

    print(f"device={device} shape=({args.B}, {args.S}, {args.N}, {args.D}) dtype={args.dtype}")

    x, phi, alpha, base = _make_inputs(args.B, args.S, args.N, args.D, dtype, device)

    out = mhc_pre_clamp_sinkhorn(
        x,
        phi,
        alpha,
        base,
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        iter_times=args.iter_times,
        need_backward=args.need_backward,
    )
    _sync(device)

    print("run ok")
    print(f"y.shape={tuple(out['y'].shape)} dtype={out['y'].dtype}")
    print(f"post_out.shape={tuple(out['post_out'].shape)} dtype={out['post_out'].dtype}")
    print(f"comb_frag.shape={tuple(out['comb_frag'].shape)} dtype={out['comb_frag'].dtype}")
    if args.need_backward:
        for key in ["inv_rms", "x_scaled", "mixes", "h_res_logits", "pre"]:
            print(f"{key}.shape={tuple(out[key].shape)} dtype={out[key].dtype}")

    if args.check:
        ref = mhc_pre_clamp_sinkhorn_ref(
            x,
            phi,
            alpha,
            base,
            norm_eps=args.norm_eps,
            hc_eps=args.hc_eps,
            clamp_min=args.clamp_min,
            clamp_max=args.clamp_max,
            iter_times=args.iter_times,
        )
        y_ok = torch.allclose(out["y"].float(), ref["y"].float(), atol=1e-2, rtol=1e-2)
        post_ok = torch.allclose(out["post_out"].float(), ref["post_out"].float(), atol=1e-4, rtol=1e-4)
        comb_ok = torch.allclose(out["comb_frag"].float(), ref["comb_frag"].float(), atol=1e-4, rtol=1e-4)
        print(f"check y={y_ok} post_out={post_ok} comb_frag={comb_ok}")
        if not (y_ok and post_ok and comb_ok):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
