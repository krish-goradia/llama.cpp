#!/usr/bin/env python3
"""GPU busy/idle analysis from an Nsight Systems report.

Answers: while the server was decoding, what fraction of time was the GPU
actually executing work, and how large are the idle gaps between steps?

Usage:
    python3 gpu_gap_analysis.py profiles/nsys_phase2_123.nsys-rep
    python3 gpu_gap_analysis.py profiles/nsys_phase2_123.sqlite --bubble-us 100

Notes:
  * All GPU activity (kernels, memcpy, memset, graph launches) is merged into
    one timeline, so overlapping streams are not double counted.
  * Gaps longer than --max-gap-ms are treated as "server idle" (between request
    waves, before/after the load) and excluded from the window.
  * --bubble-us is a heuristic cut-off: gaps shorter than it are assumed to be
    kernel-launch latency inside a step, longer gaps are assumed to be CPU work
    between steps. Check the histogram before trusting that split.
  * If the server uses CUDA graphs, add --cuda-graph-trace=node to the nsys
    command for per-kernel visibility (graph-level tracing is also handled).
"""
import argparse
import os
import sqlite3
import subprocess
import sys

TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
)
# (low_us, high_us) buckets for the gap histogram
BUCKETS = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, 5000), (5000, float("inf"))]


def export_sqlite(rep):
    db = os.path.splitext(rep)[0] + ".sqlite"
    if not os.path.exists(db):
        subprocess.check_call(
            ["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", db, rep]
        )
    return db


def load_intervals(db):
    con = sqlite3.connect(db)
    present = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    intervals, used = [], []
    for t in TABLES:
        if t in present:
            rows = con.execute(f'select start, "end" from {t}').fetchall()
            if rows:
                intervals += rows
                used.append(f"{t.replace('CUPTI_ACTIVITY_KIND_', '')}={len(rows)}")
    con.close()
    intervals.sort()
    return intervals, used


def merge(iv):
    out = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help=".nsys-rep or .sqlite file")
    ap.add_argument("--max-gap-ms", type=float, default=50.0)
    ap.add_argument("--bubble-us", type=float, default=50.0)
    a = ap.parse_args()

    db = a.report if a.report.endswith(".sqlite") else export_sqlite(a.report)
    iv, used = load_intervals(db)
    if not iv:
        sys.exit("No GPU activity found. Was the capture started (nsys start) before the load?")

    m = merge(iv)
    busy = sum(e - s for s, e in m)
    gaps = [m[i + 1][0] - m[i][1] for i in range(len(m) - 1)]
    max_gap_ns = a.max_gap_ms * 1e6
    idle_excluded = sum(g for g in gaps if g > max_gap_ns)
    gaps_in = [g for g in gaps if g <= max_gap_ns]
    window = busy + sum(gaps_in)

    print(f"Source tables      : {', '.join(used)}")
    print(f"Merged intervals   : {len(m)}")
    print(f"Window             : {window / 1e9:.3f} s  (excluded idle > {a.max_gap_ms} ms: {idle_excluded / 1e9:.3f} s)")
    print(f"GPU busy           : {100 * busy / window:.1f}%")
    print()
    print(f"{'gap bucket (us)':>18} {'count':>9} {'total ms':>10} {'% window':>9}")
    for lo, hi in BUCKETS:
        sel = [g for g in gaps_in if lo * 1e3 <= g < hi * 1e3]
        label = f"{lo}-{hi}" if hi != float("inf") else f">={lo}"
        tot = sum(sel)
        print(f"{label:>18} {len(sel):>9} {tot / 1e6:>10.1f} {100 * tot / window:>8.1f}%")

    bubble_ns = a.bubble_us * 1e3
    bubbles = [g for g in gaps_in if g >= bubble_ns]
    bub_tot = sum(bubbles)
    print()
    print(f"Bubbles >= {a.bubble_us:g} us : {len(bubbles)} gaps, {100 * bub_tot / window:.1f}% of window")
    if bubbles:
        print(f"  mean bubble       : {bub_tot / len(bubbles) / 1e3:.0f} us")
        print(f"  implied step time : {window / (len(bubbles) + 1) / 1e6:.2f} ms (if each bubble is a step boundary)")
    print()
    print("Reading it: the bubble share is an upper bound on what CPU/GPU overlap could recover.")
    print("  < 5%  -> overlap is not worth building.   10-20% -> worth building.")


if __name__ == "__main__":
    main()
