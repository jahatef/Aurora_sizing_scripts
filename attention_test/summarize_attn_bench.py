#!/usr/bin/env python
"""Merge bench_attn_kernels.py CSVs and print one comparison table per shape.

usage: python summarize_attn_bench.py results/*.csv
"""
import csv
import sys
from collections import defaultdict


def main(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))

    gemm = [r for r in rows if r["kernel"] == "gemm" and r["status"] == "ok"]
    if gemm:
        print("Dense bf16 GEMM reference: " + ", ".join(
            f"{r['seqlen']}^3 {float(r['fwd_tflops']):.0f} TF" for r in gemm))
        peak = max(float(r["fwd_tflops"]) for r in gemm)
    else:
        peak = None

    unavailable = [r for r in rows if r["status"] == "unavailable"]
    for r in unavailable:
        print(f"[unavailable] {r['kernel']}: {r['note']}")

    by_shape = defaultdict(list)
    for r in rows:
        if r["kernel"] == "gemm" or r["status"] == "unavailable":
            continue
        by_shape[(int(r["heads"]), int(r["head_dim"]), int(r["batch"]), int(r["seqlen"]))].append(r)

    for (h, d, b, s), rs in sorted(by_shape.items()):
        print(f"\n== heads {h} x head_dim {d} | batch {b} | local seqlen {s}")
        print(f"   {'kernel':<20s} {'fwd ms':>9s} {'fwd TF':>7s} {'f+b ms':>9s} {'f+b TF':>7s}"
              f" {'% gemm':>7s}  {'max rel err':>11s}  status")
        ok = [r for r in rs if r["status"] == "ok"]
        best = max(ok, key=lambda r: float(r["fwd_bwd_tflops"]), default=None)
        for r in sorted(rs, key=lambda r: -float(r["fwd_bwd_tflops"] or 0)):
            label = r["kernel"] + (f" [{r['backend']}]" if r["backend"] else "")
            errs = [float(r[k]) for k in ("err_out", "err_dq", "err_dk", "err_dv") if r.get(k)]
            err = f"{max(errs):.1e}" if errs else "-"
            pct = (f"{100*float(r['fwd_bwd_tflops'])/peak:.0f}%"
                   if peak and r["fwd_bwd_tflops"] else "-")
            mark = " <- best" if r is best else ""
            if errs and max(errs) > 5e-2:
                mark += " (WRONG?)"
            print(f"   {label:<20s} {r['fwd_ms'] or '-':>9s} {r['fwd_tflops'] or '-':>7s}"
                  f" {r['fwd_bwd_ms'] or '-':>9s} {r['fwd_bwd_tflops'] or '-':>7s} {pct:>7s}"
                  f"  {err:>11s}  {r['status']}{mark} {r['note'] if r['status'] != 'ok' else ''}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
