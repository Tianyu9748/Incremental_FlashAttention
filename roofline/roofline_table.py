#!/usr/bin/env python3
"""Roofline / compute-vs-bandwidth utilization table for the PRR hardware-dependency answer.

Parses the Nsight Compute per-stage profiles already committed under
``latency/triton/ncu_out_<method>_bsr_GLM-4-9B-1M_<ctx>/result.txt`` (GLM-4-9B, batch 1,
Quest and InfLLM-v2, context lengths 16K..512K) and produces:

1. ``utilization.csv`` — tidy rows of (method, context, stage, dur_us, dur_pct, sm_pct,
   l2_pct, dram_pct).
2. A printed headroom table showing that decode leaves the SMs almost entirely idle
   (idle-compute headroom = 100 - SM%), which is the resource PRR's speculative attention
   consumes.
3. A first-order projection of how that idle headroom grows as future hardware widens the
   compute:bandwidth ratio (peak FLOPs growing faster than HBM bandwidth).

This is a reproducible companion to the paper's Figure 4 / Table 6; it introduces no new
measurements, only re-aggregates the existing profiles.

Usage:
    python rebuttal/q3_roofline/roofline_table.py
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

# The Nsight Compute profiles live in the DynamicSparseAttn repo (not this one). Point
# PRR_NCU_ROOT at ``<DynamicSparseAttn>/latency/triton`` to re-run; otherwise the committed
# ``utilization.csv`` already holds the aggregated numbers used in the paper.
NCU_ROOT = Path(os.environ.get("PRR_NCU_ROOT", "/home/wangtian/DynamicSparseAttn/latency/triton"))
OUT_CSV = Path(__file__).resolve().parent / "utilization.csv"

DIR_RE = re.compile(r"ncu_out_(?P<method>\w+?)_bsr_GLM-4-9B-1M_(?P<ctx>\d+K)$")
ROW_RE = re.compile(
    r"^\s*(?P<stage>[A-Za-z_]+)\s+(?P<kernels>\d+)\s+(?P<dur>[\d.]+)\s+(?P<durpct>[\d.]+)\s+"
    r"(?P<sm>[\d.]+)\s+(?P<l2>[\d.]+)\s+(?P<dram>[\d.]+)\s*$"
)

# Table 6 (paper appendix): GLM-4-9B utilization (%) under a batch-size sweep at 128K
# context. Reproduced here because it is not part of the NCU context-length profiles but
# reinforces that the SMs stay idle even at larger batch sizes.
TABLE6_BATCH = {
    "quest": {2: (6.73, 43.46, 36.55), 4: (7.22, 44.33, 37.26), 8: (7.90, 44.32, 37.24), 16: (8.66, 45.57, 38.02)},
    "infllm": {2: (6.55, 42.53, 35.73), 4: (7.31, 43.90, 36.71), 8: (7.93, 45.84, 38.40), 16: (9.27, 47.36, 39.50)},
}

CTX_ORDER = {"16K": 16, "32K": 32, "64K": 64, "128K": 128, "256K": 256, "512K": 512}


@dataclass(frozen=True)
class StageRow:
    """One profiled stage of a single (method, context) decode layer."""

    method: str
    ctx: str
    stage: str
    dur_us: float
    dur_pct: float
    sm_pct: float
    l2_pct: float
    dram_pct: float


def parse_result(path: Path, method: str, ctx: str) -> list[StageRow]:
    """Parse one ``result.txt`` into per-stage rows.

    Args:
        path: Path to a ``result.txt`` produced by the NCU profiling scripts.
        method: Selection method label (``quest`` or ``infllm``).
        ctx: Context-length label (e.g. ``128K``).

    Returns:
        The parsed stage rows (including the pooled ``LAYER`` row).
    """
    rows: list[StageRow] = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        rows.append(
            StageRow(
                method=method,
                ctx=ctx,
                stage=m["stage"],
                dur_us=float(m["dur"]),
                dur_pct=float(m["durpct"]),
                sm_pct=float(m["sm"]),
                l2_pct=float(m["l2"]),
                dram_pct=float(m["dram"]),
            )
        )
    return rows


def collect() -> list[StageRow]:
    """Parse every ``ncu_out_*`` profile under ``latency/triton``."""
    out: list[StageRow] = []
    for d in sorted(NCU_ROOT.glob("ncu_out_*_bsr_GLM-4-9B-1M_*")):
        m = DIR_RE.match(d.name)
        rf = d / "result.txt"
        if not m or not rf.exists():
            continue
        out.extend(parse_result(rf, m["method"], m["ctx"]))
    return out


def write_csv(rows: list[StageRow]) -> None:
    """Write the tidy utilization table to ``utilization.csv``."""
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "context", "context_k", "stage", "dur_us", "dur_pct", "sm_pct", "l2_pct", "dram_pct"])
        for r in sorted(rows, key=lambda x: (x.method, CTX_ORDER.get(x.ctx, 0), x.stage)):
            w.writerow([r.method, r.ctx, CTX_ORDER.get(r.ctx, 0), r.stage,
                        f"{r.dur_us:.2f}", f"{r.dur_pct:.1f}", f"{r.sm_pct:.2f}", f"{r.l2_pct:.2f}", f"{r.dram_pct:.2f}"])


def print_headroom(rows: list[StageRow]) -> None:
    """Print the idle-compute headroom per (method, context) for the pooled layer."""
    layer = [r for r in rows if r.stage == "LAYER"]
    print("\n== Per-layer utilization and idle-compute headroom (GLM-4-9B, batch 1) ==")
    print(f"{'method':<8}{'ctx':>6}{'SM%':>8}{'L2-BW%':>9}{'DRAM-BW%':>10}{'idle SM%':>10}")
    print("-" * 51)
    for r in sorted(layer, key=lambda x: (x.method, CTX_ORDER.get(x.ctx, 0))):
        print(f"{r.method:<8}{r.ctx:>6}{r.sm_pct:>8.2f}{r.l2_pct:>9.2f}{r.dram_pct:>10.2f}{100 - r.sm_pct:>10.2f}")
    if layer:
        avg_sm = sum(r.sm_pct for r in layer) / len(layer)
        avg_dram = sum(r.dram_pct for r in layer) / len(layer)
        print("-" * 51)
        print(f"{'mean':<8}{'':>6}{avg_sm:>8.2f}{'':>9}{avg_dram:>10.2f}{100 - avg_sm:>10.2f}")
        print(f"\n  -> decode uses ~{avg_sm:.0f}% of peak compute; ~{100 - avg_sm:.0f}% of the SMs are idle,")
        print("     which is the headroom PRR fills with speculative attention.")


def print_table6() -> None:
    """Print the Table 6 batch-size sweep (from the paper) for context."""
    print("\n== Batch-size sweep at 128K (paper Table 6): SMs stay idle even at BS=16 ==")
    print(f"{'method':<8}{'BS':>4}{'SM%':>8}{'L2-BW%':>9}{'DRAM-BW%':>10}{'idle SM%':>10}")
    print("-" * 49)
    for method, sweep in TABLE6_BATCH.items():
        for bs, (sm, l2, dram) in sweep.items():
            print(f"{method:<8}{bs:>4}{sm:>8.2f}{l2:>9.2f}{dram:>10.2f}{100 - sm:>10.2f}")


def print_projection(rows: list[StageRow]) -> None:
    """Project idle-compute headroom under a widening compute:bandwidth ratio.

    First-order model: long-context decode is memory-bound, so the wall-clock time of the
    memory-bound stages is set by bytes-moved / HBM-bandwidth. If a future GPU multiplies
    peak compute throughput by ``k`` while bandwidth (hence the memory-bound runtime) grows
    more slowly, the *same* attention work occupies SM% / k of peak compute. Idle-compute
    headroom therefore grows toward 100% as k increases. This is exactly the regime PRR
    benefits from; the projection is first-order (ignores second-order effects such as the
    selection stage itself speeding up) and is meant to illustrate the direction of the trend.
    """
    layer = [r for r in rows if r.stage == "LAYER"]
    if not layer:
        return
    avg_sm = sum(r.sm_pct for r in layer) / len(layer)
    print("\n== Projection: idle-compute headroom vs compute:bandwidth widening ==")
    print("   (k = future peak-compute growth relative to HBM bandwidth; memory-bound runtime held)")
    print(f"{'k (compute/BW)':>16}{'SM% used':>12}{'idle SM%':>12}")
    print("-" * 40)
    for k in (1, 2, 4, 8):
        used = avg_sm / k
        print(f"{k:>16}{used:>12.2f}{100 - used:>12.2f}")
    print("\n  -> As peak compute outpaces bandwidth (mainstream trend + fp8/fp4 tensor cores),")
    print("     the idle compute PRR exploits grows, so PRR's overlap opportunity is robust to")
    print("     future hardware. It would only shrink in a bandwidth-dominant regime that also")
    print("     removes the selection-to-attention serialization.")


def main() -> None:
    rows = collect()
    if not rows:
        raise SystemExit(f"no NCU profiles found under {NCU_ROOT}")
    write_csv(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} stage rows across "
          f"{len({(r.method, r.ctx) for r in rows})} (method, context) profiles)")
    print_headroom(rows)
    print_table6()
    print_projection(rows)


if __name__ == "__main__":
    main()
