#!/usr/bin/env python3
"""Score a Stage-2 out_root AND read that score in DISTINCT SCENES (seeds can collapse onto one scene).

Scene fingerprint = every float of the program's first perception printout in logs/<tag>_<seed>.log,
rounded to 4 dp; programs that print nothing usable report "not measurable".

Usage: stage2_report.py <out_root> [--json]
"""
import json
import os
import re
import sys

# numpy may print scientific notation ("-2.595e-01"): the exponent must be part of the token.
FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")
# A bracketed float vector, comma- or whitespace-separated (list or bare numpy repr).
VECLINE = re.compile(r"\[\s*-?\d+\.\d+(?:[eE][-+]?\d+)?[\s,]")
# Lines that carry floats but describe MOTION, not the initial scene.
NOISE = re.compile(
    r"Running Trials|it/s|Saved interaction|took .* seconds|WARNING|Warp |"
    r"warn\(|UserWarning|sapien|robosuite|Traceback|File \"", re.I)


def trials(out_root, tag):
    d = os.path.join(out_root, "oracle", tag)
    out = {}
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        m = re.match(r"trial_(\d+)_.*reward_([\d.]+)_", name)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def fingerprint(out_root, tag, seed):
    """First perception printout of the run, as a rounded float tuple."""
    path = os.path.join(out_root, "logs", "%s_%d.log" % (tag, seed))
    if not os.path.isfile(path):
        return None
    with open(path, errors="replace") as fh:
        for line in fh:
            # tqdm writes the bar onto the same line as the first print
            line = line.split("it/s]")[-1]
            if NOISE.search(line) or not VECLINE.search(line):
                continue
            vals = [round(float(v), 4) for v in FLOAT.findall(line)]
            if len(vals) >= 2:
                return tuple(vals)
    return None


def main():
    out_root = sys.argv[1].rstrip("/")
    rows, scenes, unmeasured = [], {}, 0
    for tag in ("clean", "randomized"):
        for seed, reward in sorted(trials(out_root, tag).items()):
            fp = fingerprint(out_root, tag, seed)
            if fp is None:
                unmeasured += 1
            else:
                scenes.setdefault((tag, fp), []).append(seed)
            rows.append(dict(tag=tag, seed=seed, reward=reward, fp=fp))

    clean = [r for r in rows if r["tag"] == "clean"]
    rand = [r for r in rows if r["tag"] == "randomized"]
    c = sum(r["reward"] >= 1.0 for r in clean)
    r_ = sum(r["reward"] >= 1.0 for r in rand)
    n = c + r_
    tot = len(rows)

    if unmeasured == tot:
        scene_str = "not measurable"
    else:
        scene_str = "%d%s" % (len(scenes),
                              "" if not unmeasured else
                              " (+%d seed(s) not measurable)" % unmeasured)

    print("out_root: %s" % out_root)
    print("score:    clean %d/%d + randomized %d/%d = %d/%d  [bar >=12/15]  %s"
          % (c, len(clean), r_, len(rand), n, tot,
             "PASS" if n >= 12 else "FAIL"))
    print("randomized half (predicts collection yield): %d/%d" % (r_, len(rand)))
    print("distinct scenes: %s" % scene_str)
    if len(scenes) < tot - unmeasured:
        print("collapsed groups (>1 seed on one scene):")
        for (tag, fp), seeds in sorted(scenes.items(), key=lambda kv: kv[1]):
            if len(seeds) > 1:
                print("  %-11s %s -> one scene" % (tag, seeds))
    print("\nper-seed:")
    for r in rows:
        print("  %-11s %d  reward %.3f  %s  scene=%s"
              % (r["tag"], r["seed"], r["reward"],
                 "PASS" if r["reward"] >= 1.0 else "FAIL",
                 "?" if r["fp"] is None else str(r["fp"])[:70]))
    if "--json" in sys.argv:
        print(json.dumps(dict(clean=c, rand=r_, n=n, tot=tot,
                              scenes=len(scenes), unmeasured=unmeasured)))


if __name__ == "__main__":
    main()
