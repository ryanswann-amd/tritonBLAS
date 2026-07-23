#!/usr/bin/env python3
"""Aggregate sweep_tune CSV logs -> per-shape best steal-k config + comparison.

Reads all CSV rows (lines starting 'CSV,') from the given files, and for each
shape prints: library dp/sk, the best correct steal-k config, and steal-k's
speedup vs dp and sk. Also writes a tidy combined CSV to data/raw/.

Usage: python3 aggregate_sweep.py sweep_*.log
"""
import sys, csv, glob

COLS = ["shape", "m", "n", "k", "variant", "bm", "bn", "bk", "path", "grid", "ns", "nwarps", "ms", "tflops", "ok", "vs_dp", "vs_sk"]


def load(files):
    rows = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line.startswith("CSV,"):
                continue
            parts = line[4:].split(",")
            if len(parts) != len(COLS):
                continue
            rows.append(dict(zip(COLS, parts)))
    return rows


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    files = []
    for a in sys.argv[1:]:
        files.extend(glob.glob(a))
    rows = load(files)
    if not rows:
        print("no CSV rows found")
        return

    # group by shape, preserve first-seen order
    shapes = []
    by_shape = {}
    for r in rows:
        s = r["shape"]
        if s not in by_shape:
            by_shape[s] = []
            shapes.append(s)
        by_shape[s].append(r)

    # tidy combined CSV
    import os
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/sweep_tune_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def keymnk(s):
        m, n, k = (int(x) for x in s.split("x"))
        return (m * n * k, m, n, k)
    shapes.sort(key=keymnk)

    hdr = f"{'shape':>18} | {'dp TF/s':>8} {'sk TF/s':>8} | {'best steal':>10} {'tile':>11} {'path':>8} {'grid':>4} {'ns':>2} | {'vsDP':>5} {'vsSK':>5}"
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for s in shapes:
        rs = by_shape[s]
        dp = next((r for r in rs if r["variant"] == "dp"), None)
        sk = next((r for r in rs if r["variant"] == "sk"), None)
        steals = [r for r in rs if r["variant"] == "steal" and r["ok"] == "True" and fnum(r["tflops"]) is not None]
        if not steals:
            print(f"{s:>18} | (no correct steal-k config)")
            continue
        best = max(steals, key=lambda r: fnum(r["tflops"]))
        dp_tf = dp["tflops"] if dp else "-"
        sk_tf = sk["tflops"] if sk else "-"
        tile = f"{best['bm']}x{best['bn']}x{best['bk']}"
        print(f"{s:>18} | {dp_tf:>8} {sk_tf:>8} | {best['tflops']:>10} {tile:>11} {best['path']:>8} {best['grid']:>4} {best['ns']:>2} | {best['vs_dp']:>5} {best['vs_sk']:>5}")
        summary.append((s, dp_tf, sk_tf, best))

    # count wins
    wins_sk = sum(1 for (_, _, _, b) in summary if fnum(b["vs_sk"]) and fnum(b["vs_sk"]) >= 1.0)
    wins_dp = sum(1 for (_, _, _, b) in summary if fnum(b["vs_dp"]) and fnum(b["vs_dp"]) >= 1.0)
    print(f"\nsteal-k best >= stream-k on {wins_sk}/{len(summary)} shapes; >= data-parallel on {wins_dp}/{len(summary)}.")
    print("wrote data/raw/sweep_tune_all.csv")


if __name__ == "__main__":
    main()
