#!/usr/bin/env python
"""
Single-tile causal attention kernel benchmark.

Compares the per-tile attention kernels that matter for GPT-NeoX + ringX on
Aurora (PVC), with no ring communication involved:

  ringX1/2/3/4  ringX_attn.ringX{N}_attn_func on a 1-rank process group
                (backend chosen by RINGX_ATTN_BACKEND, set per run with
                --ringx-backend; launch one process per backend)
  sdpa          torch.nn.functional.scaled_dot_product_attention(is_causal=True)
  flash_attn    flash_attn.flash_attn_func, if an XPU/CUDA build is installed
  eager         QK^T -> masked softmax -> PV in plain matmuls (reference;
                skipped automatically when the score matrix is too large)
  gemm          a dense bf16 matmul, to anchor what "good" looks like on the tile

For each (kernel, batch, seqlen, heads x head_dim) it times forward and
forward+backward separately, reports causal-attention TFLOPS, and checks the
output and q/k/v gradients against a reference so a fast-but-wrong kernel
cannot win.

FLOP model (causal, so half of the full s x s):
  fwd     = 2 * b * h * s^2 * d        (two matmuls, x2 FLOPs/MAC, /2 causal)
  bwd     = 2.5 * fwd                  (5 matmuls vs 2)
  fwd_bwd = 3.5 * fwd

Example (one PVC tile):
  RINGX_ATTN_BACKEND=fused python bench_attn_kernels.py \
      --kernels ringX3,sdpa,flash_attn,eager \
      --batch 4,28 --seqlen 2048,4096,8192 --head-configs 32x64,16x128 \
      --csv results_fused.csv
"""

import argparse
import csv
import importlib
import math
import os
import sys
import tempfile
import time
import traceback

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# device helpers
# --------------------------------------------------------------------------

def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device):
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def is_oom(exc):
    if hasattr(torch, "OutOfMemoryError") and isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg or "out_of_device_memory" in msg
            or "out_of_resources" in msg)


# --------------------------------------------------------------------------
# kernels: each returns (layout, fn) where layout is "bshd" or "bhsd" and
# fn(q, k, v) -> out in the same layout
# --------------------------------------------------------------------------

_PG_READY = False


def ensure_single_rank_group(device):
    """ringX wants a process group; give it a 1-rank one."""
    global _PG_READY
    import torch.distributed as dist
    if dist.is_initialized():
        _PG_READY = True
        return dist.group.WORLD
    backend = "gloo"
    if device.type == "xpu":
        backend = "xccl"
    elif device.type == "cuda":
        backend = "nccl"
    init_file = tempfile.NamedTemporaryFile(prefix="attnbench_pg_", delete=False).name
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    try:
        dist.init_process_group(backend=backend, init_method=f"file://{init_file}",
                                rank=0, world_size=1)
    except Exception:
        # some stacks name the XPU backend "ccl" (oneCCL bindings)
        if device.type == "xpu":
            try:
                import oneccl_bindings_for_pytorch  # noqa: F401
            except ImportError:
                pass
            dist.init_process_group(backend="ccl", init_method=f"file://{init_file}",
                                    rank=0, world_size=1)
        else:
            raise
    _PG_READY = True
    return dist.group.WORLD


def make_ringx(name, device):
    mod = importlib.import_module(f"ringX_attn.{name}_attn")
    fn = getattr(mod, f"{name}_attn_func")
    group = ensure_single_rank_group(device)

    def run(q, k, v):
        # same call NeoX makes in ParallelSelfAttention.ring_attention
        return fn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True,
                  window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                  return_attn_probs=False, group=group)

    return "bshd", run


def make_sdpa(device):
    def run(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return "bhsd", run


def make_flash_attn(device):
    from flash_attn import flash_attn_func

    def run(q, k, v):
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    return "bshd", run


def make_eager(device):
    def run(q, k, v):
        s = q.shape[-2]
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        mask = torch.ones(s, s, dtype=torch.bool, device=q.device).triu_(1)
        scores = scores.masked_fill(mask, float("-inf"))
        p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(p, v)
    return "bhsd", run


def build_kernel(name, device):
    if name.startswith("ringX"):
        return make_ringx(name, device)
    if name == "sdpa":
        return make_sdpa(device)
    if name == "flash_attn":
        return make_flash_attn(device)
    if name == "eager":
        return make_eager(device)
    raise ValueError(f"unknown kernel {name}")


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def time_it(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters


def causal_fwd_flops(b, h, s, d):
    return 2.0 * b * h * s * s * d


def make_qkv(layout, b, s, h, d, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (b, s, h, d) if layout == "bshd" else (b, h, s, d)
    return [torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
            .requires_grad_(True) for _ in range(3)]


def to_bhsd(t, layout):
    return t if layout == "bhsd" else t.transpose(1, 2)


def reference_check(run, layout, b, s, h, d, dtype, device, ref_seq_cap):
    """Compare out and dq/dk/dv against an fp32 eager reference on a slice
    small enough to materialise the score matrix."""
    b_c, s_c = min(b, 1), min(s, ref_seq_cap)
    q, k, v = make_qkv(layout, b_c, s_c, h, d, dtype, device, seed=1)
    out = run(q, k, v)
    go = torch.randn_like(out)
    out.backward(go)

    qr, kr, vr = [to_bhsd(t.detach(), layout).float().requires_grad_(True) for t in (q, k, v)]
    _, eager = make_eager(device)
    out_r = eager(qr, kr, vr)
    out_r.backward(to_bhsd(go, layout).float())

    def rel(a, ref):
        a = to_bhsd(a, layout).float()
        return ((a - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()

    return {
        "err_out": rel(out.detach(), out_r.detach()),
        "err_dq": rel(q.grad, qr.grad),
        "err_dk": rel(k.grad, kr.grad),
        "err_dv": rel(v.grad, vr.grad),
    }


def bench_one(name, run, layout, b, s, h, d, dtype, device, args):
    row = dict(kernel=name, backend=os.environ.get("RINGX_ATTN_BACKEND", "") if name.startswith("ringX") else "",
               batch=b, seqlen=s, heads=h, head_dim=d, layout=layout,
               status="ok", fwd_ms="", fwd_bwd_ms="", fwd_tflops="", fwd_bwd_tflops="",
               err_out="", err_dq="", err_dk="", err_dv="", note="")

    if name == "eager":
        score_bytes = b * h * s * s * 4  # fp32 softmax copy dominates
        if score_bytes > args.eager_max_gb * 1e9:
            row.update(status="skipped", note=f"scores {score_bytes/1e9:.0f} GB > --eager-max-gb")
            return row

    if name in ("ringX3", "ringX4"):
        if s % 2:
            row.update(status="skipped", note="ringX3/4 need even seqlen")
            return row

    try:
        if not args.skip_check:
            row.update({k: f"{v:.2e}" for k, v in
                        reference_check(run, layout, b, s, h, d, dtype, device, args.ref_seq_cap).items()})
            empty_cache(device)

        q, k, v = make_qkv(layout, b, s, h, d, dtype, device)
        flops = causal_fwd_flops(b, h, s, d)

        def fwd():
            with torch.no_grad():
                run(q, k, v)

        out = run(q, k, v)
        go = torch.randn_like(out)
        del out

        def fwd_bwd():
            q.grad = k.grad = v.grad = None
            o = run(q, k, v)
            o.backward(go)

        t_f = time_it(fwd, device, args.warmup, args.iters)
        t_fb = time_it(fwd_bwd, device, args.warmup, args.iters)
        row.update(fwd_ms=f"{t_f*1e3:.3f}", fwd_bwd_ms=f"{t_fb*1e3:.3f}",
                   fwd_tflops=f"{flops/t_f/1e12:.2f}", fwd_bwd_tflops=f"{3.5*flops/t_fb/1e12:.2f}")
    except Exception as e:  # keep the sweep going
        row.update(status="oom" if is_oom(e) else "error", note=str(e).splitlines()[0][:200] if str(e) else type(e).__name__)
        if args.verbose and row["status"] == "error":
            traceback.print_exc()
    finally:
        empty_cache(device)
    return row


def bench_gemm(dtype, device, args):
    rows = []
    for n in (4096, 8192):
        try:
            a = torch.randn(n, n, dtype=dtype, device=device)
            bm = torch.randn(n, n, dtype=dtype, device=device)
            t = time_it(lambda: a @ bm, device, args.warmup, args.iters)
            rows.append(dict(kernel="gemm", backend="", batch=1, seqlen=n, heads="", head_dim=n,
                             layout="", status="ok", fwd_ms=f"{t*1e3:.3f}", fwd_bwd_ms="",
                             fwd_tflops=f"{2*n**3/t/1e12:.2f}", fwd_bwd_tflops="",
                             err_out="", err_dq="", err_dk="", err_dv="", note=f"{n}^3 matmul"))
        except Exception as e:
            rows.append(dict(kernel="gemm", seqlen=n, status="error", note=str(e)[:200]))
        empty_cache(device)
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

FIELDS = ["kernel", "backend", "batch", "seqlen", "heads", "head_dim", "layout", "status",
          "fwd_ms", "fwd_bwd_ms", "fwd_tflops", "fwd_bwd_tflops",
          "err_out", "err_dq", "err_dk", "err_dv", "note"]


def parse_list(s, typ=int):
    return [typ(x) for x in s.split(",") if x]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernels", default="ringX3,sdpa,flash_attn,eager",
                   help="comma list from ringX1,ringX2,ringX3,ringX4,sdpa,flash_attn,eager")
    p.add_argument("--ringx-backend", default=None,
                   help="sets RINGX_ATTN_BACKEND before importing ringX (fused|portable); "
                        "ringX reads it at import, so use one value per process")
    p.add_argument("--batch", default="4,28")
    p.add_argument("--seqlen", default="2048,4096,8192",
                   help="per-tile (local) sequence lengths")
    p.add_argument("--head-configs", default="32x64,16x128",
                   help="HEADSxHEAD_DIM list; keep heads*head_dim = hidden (2048 for 1.3B)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="auto")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--no-gemm", action="store_true", help="skip the dense-GEMM reference")
    p.add_argument("--skip-check", action="store_true", help="skip the correctness check")
    p.add_argument("--ref-seq-cap", type=int, default=2048,
                   help="max seqlen used for the fp32 correctness reference")
    p.add_argument("--eager-max-gb", type=float, default=16.0,
                   help="skip eager when its fp32 score matrix exceeds this")
    p.add_argument("--csv", default="attn_kernel_bench.csv")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.ringx_backend:
        os.environ["RINGX_ATTN_BACKEND"] = args.ringx_backend

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "xpu":
        torch.xpu.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    heads_dims = [tuple(int(x) for x in hd.split("x")) for hd in args.head_configs.split(",")]
    batches, seqlens = parse_list(args.batch), parse_list(args.seqlen)

    print(f"device={device} dtype={args.dtype} torch={torch.__version__} "
          f"RINGX_ATTN_BACKEND={os.environ.get('RINGX_ATTN_BACKEND', '(unset)')}", flush=True)
    print("note: the first ringX shape can spend many minutes in Triton autotuning "
          "(set TRITON_CACHE_DIR / TRITON_AUTOTUNE_CACHE_DIR to reuse results)", flush=True)

    kernels = {}
    rows = []
    for name in parse_list(args.kernels, str):
        try:
            kernels[name] = build_kernel(name, device)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(kernel=name, status="unavailable", note=f"{type(e).__name__}: {e}"[:200]))

    if not args.no_gemm and device.type != "cpu":
        rows += bench_gemm(dtype, device, args)

    for (h, d) in heads_dims:
        for b in batches:
            for s in seqlens:
                for name, (layout, run) in kernels.items():
                    t0 = time.time()
                    row = bench_one(name, run, layout, b, s, h, d, dtype, device, args)
                    rows.append(row)
                    print(f"{name:10s} {row['backend']:8s} b={b:<3d} s={s:<6d} {h}x{d:<4d} "
                          f"{row['status']:8s} fwd {row['fwd_ms'] or '-':>9s} ms "
                          f"({row['fwd_tflops'] or '-':>6s} TF)  fwd+bwd {row['fwd_bwd_ms'] or '-':>9s} ms "
                          f"({row['fwd_bwd_tflops'] or '-':>6s} TF)  err out/dq {row['err_out'] or '-'}/{row['err_dq'] or '-'}"
                          f"  [{time.time()-t0:.0f}s] {row['note']}", flush=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nwrote {args.csv}", flush=True)

    # correctness summary
    bad = [r for r in rows if r.get("err_dq") and max(float(r[k]) for k in ("err_out", "err_dq", "err_dk", "err_dv")) > 5e-2]
    if bad:
        print("\nWARNING: these results differ from the fp32 reference by >5% (max rel error):")
        for r in bad:
            print(f"  {r['kernel']} b={r['batch']} s={r['seqlen']} {r['heads']}x{r['head_dim']}: "
                  f"out {r['err_out']} dq {r['err_dq']} dk {r['err_dk']} dv {r['err_dv']}")

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
