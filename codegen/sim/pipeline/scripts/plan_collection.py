#!/usr/bin/env python3
"""Size a multi-task collection from the frozen programs' Stage-2 scores.

    python3 pipeline/scripts/plan_collection.py [--kept 50] [--config demo_clean|demo_randomized] [--hosts h1 h2 ...] [--tasks t ...]

Prints, per task, the seeds to run for `--kept` expected successes (`ceil(kept / rate) + margin`, rate floored at
0.25) from the Stage-2 rate in frozen/<task>/task.yaml; FAILED and unvalidated tasks are listed last and marked.
"""
import argparse, math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def meta(task):
    p = ROOT / "frozen" / task / "task.yaml"
    return yaml.safe_load(p.read_text()) if p.is_file() else None


def score(m, split):
    s2 = (m or {}).get("stage2")
    if not s2:
        return None, None, None
    c, cn = (int(x) for x in s2["clean"].split("/")); r, rn = (int(x) for x in s2["randomized"].split("/"))
    return (c / cn if split == "demo_clean" else r / rn), s2["total"], s2["verdict"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kept", type=int, default=50)
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--config", default="demo_clean", choices=["demo_clean", "demo_randomized"])
    ap.add_argument("--hosts", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args()
    tasks = a.tasks or sorted(p.parent.name for p in (ROOT / "frozen").glob("*/task.yaml"))
    rows = []
    for t in tasks:
        m = meta(t)
        rate, total, verdict = score(m, a.config)
        prog = (m or {}).get("program", "-")
        if rate is None:
            rows.append((t, prog, "-", "-", None, "unvalidated"))
            continue
        eps = math.ceil(a.kept / max(rate, 0.25)) + a.margin
        rows.append((t, prog, f"{rate:.2f}", total, eps, verdict))
    rows.sort(key=lambda r: (r[5] != "PASS", r[0]))
    print(f"{'task':28s} {'program':28s} {'rate':>5s} {'stage2':>7s} {'seeds':>5s}  verdict      host")
    for i, (t, v, rate, total, eps, verdict) in enumerate(rows):
        host = a.hosts[i % len(a.hosts)] if a.hosts else ""
        print(f"{t:28s} {v:28s} {rate:>5s} {total:>7s} {eps if eps else '-':>5}  {verdict:11s}  {host}")
    n = sum(r[4] or 0 for r in rows)
    print(f"\n{len(rows)} tasks, {n} seeds in total for {a.kept} kept episodes each ({a.config}).")


if __name__ == "__main__":
    main()
