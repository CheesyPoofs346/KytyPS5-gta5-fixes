#!/usr/bin/env python3
"""Whole-route frame-time statistics from frametimes.csv captures.

Section A of the performance audit. Percentiles are computed once over every
retained route sample. Percentiles of windows are never averaged together.

Usage:
    python tools/frametimes.py run1.csv [run2.csv ...] [--live]

--live tolerates one incomplete trailing record, for reading a capture that is still
being written. Without it, any malformed row rejects the capture.

With two or more captures it also reports the spread between runs, which is the
variance floor any claimed improvement has to clear.
"""
import csv
import sys


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    idx = int(p * (len(sorted_values) - 1) + 0.5)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def load(path, live=False):
    """Parse a capture.

    A malformed INTERIOR row is a corrupt capture and is rejected outright. Only an
    incomplete FINAL row is tolerable, and only with live=True, because a running
    capture is buffered and its last record may be half-written. Every dropped row is
    reported: silently skipping records is how a capture lies about its own sample count.
    """
    route, warmup, rows = [], [], []
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    dropped = []
    for i, row in enumerate(rows):
        bad = [k for k in ("ms", "draws", "phase") if row.get(k) in (None, "")]
        if not bad:
            continue
        is_last = (i == len(rows) - 1)
        if is_last and live:
            dropped.append(i)
            continue
        raise ValueError(
            "%s: malformed %s row %d (missing %s). An interior malformed row means the "
            "capture is corrupt; rerun it. Pass --live only for a still-running capture."
            % (path, "final" if is_last else "INTERIOR", i, ",".join(bad)))

    for i, row in enumerate(rows):
        if i in dropped:
            continue
        sample = (float(row["ms"]), int(row["draws"]))
        (route if row["phase"] == "route" else warmup).append(sample)

    if dropped:
        print("  NOTE %s: dropped %d incomplete trailing record(s) (live read)"
              % (path, len(dropped)))
    return route, warmup


def summarise(path, live=False):
    route, warmup = load(path, live)
    if len(route) < 2:
        print("%s: only %d route samples, need at least 2" % (path, len(route)))
        return None

    ms = sorted(s[0] for s in route)
    draws = sorted(s[1] for s in route)
    n = len(ms)
    mean = sum(ms) / n
    med = percentile(ms, 0.50)
    p95 = percentile(ms, 0.95)
    p99 = percentile(ms, 0.99)
    over60 = sum(1 for v in ms if v > 16.667)
    over30 = sum(1 for v in ms if v > 33.333)

    print("== %s ==" % path)
    print("  samples      route=%d warmup=%d" % (n, len(warmup)))
    print("  ms           med=%.2f p95=%.2f p99=%.2f min=%.2f max=%.2f mean=%.2f"
          % (med, p95, p99, ms[0], ms[-1], mean))
    print("  fps-equiv    med=%.1f  p95ms=%.1f  p99ms=%.1f   (rate a tail-slow frame"
          " corresponds to, NOT the percentile of FPS)"
          % (1000.0 / med if med else 0, 1000.0 / p95 if p95 else 0,
             1000.0 / p99 if p99 else 0))
    print("  draws        med=%d max=%d" % (draws[n // 2], draws[-1]))
    print("  over 16.667ms (60fps, PRIMARY)  %d/%d = %.1f%%" % (over60, n, 100.0 * over60 / n))
    print("  over 33.333ms (30fps, secondary) %d/%d = %.1f%%" % (over30, n, 100.0 * over30 / n))
    return {"path": path, "n": n, "med": med, "p95": p95, "p99": p99,
            "over60": 100.0 * over60 / n, "over30": 100.0 * over30 / n}


def main():
    args = [a for a in sys.argv[1:] if a != "--live"]
    live = "--live" in sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    runs = [r for r in (summarise(p, live) for p in args) if r]
    if len(runs) < 2:
        return 0

    print()
    print("== run-to-run spread (the variance floor a real gain must clear) ==")
    for key, label in (("med", "median ms"), ("p95", "p95 ms"), ("p99", "p99 ms"),
                       ("over60", "% over 16.667ms")):
        values = [r[key] for r in runs]
        lo, hi = min(values), max(values)
        span = hi - lo
        rel = (100.0 * span / lo) if lo else 0.0
        print("  %-16s min=%.2f max=%.2f spread=%.2f (%.1f%% of min)"
              % (label, lo, hi, span, rel))
    print()
    print("  A later A/B difference smaller than this spread is not a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
