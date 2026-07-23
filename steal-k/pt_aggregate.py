#!/usr/bin/env python3
"""Aggregate pt_sweep CSV logs: steal-k (64x64 tree) vs torch.matmul.

Reports win-rate (speedup>=1), speedup percentiles, breakdown by regime
(small-both / small-M / small-N) and K bucket, correctness, and the top wins.
Writes a combined tidy CSV to data/raw/pt_sweep_all.csv.

Usage: python3 pt_aggregate.py ptsweep_shard_*.log
"""
import sys, glob, csv, os, statistics as st

COLS = ["m", "n", "k", "torch_ms", "sk_ms", "torch_tf", "sk_tf", "speedup", "ok"]
SMALL_MAX = 512


def load(files):
    rows = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line.startswith("CSV,"):
                continue
            p = line[4:].split(",")
            if len(p) != len(COLS):
                continue
            try:
                r = dict(zip(COLS, p))
                r["m"] = int(r["m"]); r["n"] = int(r["n"]); r["k"] = int(r["k"])
                if r["speedup"] in ("-", "launchfail"):
                    continue
                r["speedup"] = float(r["speedup"])
                r["sk_tf"] = float(r["sk_tf"]); r["torch_tf"] = float(r["torch_tf"])
                rows.append(r)
            except Exception:
                pass
    return rows


def regime(r):
    ms = r["m"] <= SMALL_MAX; ns = r["n"] <= SMALL_MAX
    return "both" if (ms and ns) else ("smallM" if ms else "smallN")


def pct(a, p):
    a = sorted(a); i = min(len(a) - 1, int(p / 100 * len(a)))
    return a[i]


def summarize(rows, label):
    if not rows:
        print(f"{label:14}: (none)"); return
    sp = [r["speedup"] for r in rows]
    wins = sum(1 for x in sp if x >= 1.0)
    print(f"{label:14}: n={len(rows):5} win%={100*wins/len(rows):5.1f}  "
          f"speedup p10/p50/p90/max = {pct(sp,10):.2f}/{pct(sp,50):.2f}/{pct(sp,90):.2f}/{max(sp):.2f}  "
          f"geomean={ (2.718281828**(sum(__import__('math').log(x) for x in sp)/len(sp))):.2f}")


def main():
    files = []
    for a in sys.argv[1:]:
        files.extend(glob.glob(a))
    rows = load(files)
    if not rows:
        print("no rows"); return
    bad = [r for r in rows if r["ok"] != "True"]
    print(f"total rows={len(rows)}  correctness: {len(rows)-len(bad)} ok, {len(bad)} MISMATCH")
    print()
    summarize(rows, "ALL")
    for g in ("both", "smallM", "smallN"):
        summarize([r for r in rows if regime(r) == g], g)
    print()
    print("by K bucket:")
    for lo, hi in [(0, 4096), (4096, 16384), (16384, 65536), (65536, 10**9)]:
        summarize([r for r in rows if lo < r["k"] <= hi], f"K in ({lo},{hi}]")
    print()
    print("by min(M,N) bucket:")
    for lo, hi in [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512)]:
        summarize([r for r in rows if lo < min(r["m"], r["n"]) <= hi], f"minMN ({lo},{hi}]")
    print()
    wins = sorted([r for r in rows if r["speedup"] >= 1.0], key=lambda r: -r["speedup"])
    print(f"steal-k WINS on {len(wins)}/{len(rows)} ({100*len(wins)/len(rows):.1f}%). Top 15 wins:")
    for r in wins[:15]:
        print(f"  {r['m']}x{r['n']}x{r['k']:>6}  sk={r['sk_tf']:.1f} torch={r['torch_tf']:.1f} TF/s  speedup={r['speedup']:.2f}x")
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/pt_sweep_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLS})
    print("\nwrote data/raw/pt_sweep_all.csv")


if __name__ == "__main__":
    main()
