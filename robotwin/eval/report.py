#!/usr/bin/env python3
"""Results table for one or more RoboTwin benchmark runs (robotwin/eval/benchmark.py).

    python robotwin/eval/report.py <name> [<name> ...] [--out FILE.md]

Per task and setting: successes / episodes summed over the shards that finished, as a percentage. The average row is
the mean of the per-task success rates (how RoboTwin 2.0 reports), over the tasks with a result; the pooled rate
(all successes / all episodes) is given beside it. A cell with missing shards is marked `*` and counted in the
footer.

One run writes report.md / report.csv / summary.json into evaluate_results/robotwin_benchmark/<name>/; several runs
print the comparison and write it to --out when given.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROBOTWIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTWIN))
from hybrid_eval import config  # noqa: E402

import yaml  # noqa: E402

BENCH_ROOT = config.REPO / "evaluate_results" / "robotwin_benchmark"


def collect(name):
    """{(task, protocol): {"ok", "n", "shards", "done"}} for one benchmark run, plus its manifest."""
    mf = BENCH_ROOT / name / "manifest.json"
    if not mf.is_file():
        sys.exit(f"no benchmark named {name!r} ({mf} missing)")
    m = json.loads(mf.read_text())
    cells = {}
    for j in m["jobs"]:
        c = cells.setdefault((j["task"], j["protocol"]), {"ok": 0, "n": 0, "shards": 0, "done": 0})
        c["shards"] += 1
        try:
            r = json.loads((Path(j["results_dir"]) / "run_meta.json").read_text()).get("result") or {}
        except (OSError, ValueError):
            r = {}
        if r.get("total"):
            c["ok"] += int(r["success"])
            c["n"] += int(r["total"])
            c["done"] += 1
    return m, cells


def pct(c):
    if not c or not c["n"]:
        return None
    return 100.0 * c["ok"] / c["n"]


def fmt(c):
    p = pct(c)
    if p is None:
        return "—"
    return f"{p:.0f}%" + ("*" if c["done"] < c["shards"] else "")


def summary(cells, tasks, proto):
    rates = [pct(cells.get((t, proto))) for t in tasks]
    rates = [r for r in rates if r is not None]
    ok = sum(cells[(t, proto)]["ok"] for t in tasks if (t, proto) in cells)
    n = sum(cells[(t, proto)]["n"] for t in tasks if (t, proto) in cells)
    return {"mean_task_rate": sum(rates) / len(rates) if rates else None, "tasks_with_result": len(rates),
            "pooled_rate": 100.0 * ok / n if n else None, "episodes": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bench = yaml.safe_load((config.CFG_DIR / "benchmark.yaml").read_text(encoding="utf-8"))
    labels = {k: v.get("label", k) for k, v in bench["protocols"].items()}
    runs = [(n, *collect(n)) for n in a.names]
    tasks = sorted({t for _, m, _ in runs for t in m["tasks"]})
    protos = [p for p in bench["protocols"] if any(p in m["protocols"] for _, m, _ in runs)]

    head = ["task"] + [f"{n} ({labels[p]})" for n, m, _ in runs for p in protos if p in m["protocols"]]
    rows = []
    for t in tasks:
        row = [t] + [fmt(cells.get((t, p))) for n, m, cells in runs for p in protos if p in m["protocols"]]
        rows.append(row)
    avg = ["**average** (mean of task rates; pooled)"]
    sums = {}
    for n, m, cells in runs:
        for p in protos:
            if p in m["protocols"]:
                s = summary(cells, tasks, p)
                sums[(n, p)] = s
                avg.append("—" if s["mean_task_rate"] is None else
                           f"**{s['mean_task_rate']:.1f}%**; {s['pooled_rate']:.1f}% ({s['tasks_with_result']} tasks, {s['episodes']} eps)")

    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows] + ["| " + " | ".join(avg) + " |"]
    missing = [(n, t, p, c["done"], c["shards"]) for n, _, cells in runs for (t, p), c in sorted(cells.items())
               if c["done"] < c["shards"]]
    notes = []
    for n, m, _ in runs:
        notes.append(f"- `{n}`: preset `{m['model']}`, checkpoint `{m['ckpt']}`, mode {m['mode']}, plan {m['plan']}")
    if missing:
        notes.append(f"- `*` = incomplete: {len(missing)} task/setting cells are missing shards "
                     f"(e.g. {missing[0][1]} {missing[0][2]}: {missing[0][3]}/{missing[0][4]} shards).")
    text = "# RoboTwin 2.0 benchmark\n\n" + "\n".join(lines) + "\n\n" + "\n".join(notes) + "\n"
    print(text)

    if len(runs) == 1:
        n, m, cells = runs[0]
        d = BENCH_ROOT / n
        (d / "report.md").write_text(text)
        with open(d / "report.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["task", "protocol", "success", "episodes", "rate", "shards_done", "shards"])
            for (t, p), c in sorted(cells.items()):
                wr.writerow([t, p, c["ok"], c["n"], "" if pct(c) is None else f"{pct(c):.2f}", c["done"], c["shards"]])
        (d / "summary.json").write_text(json.dumps({p: s for (_, p), s in sums.items()}, indent=1))
        print(f"wrote {d}/report.md, report.csv, summary.json")
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
