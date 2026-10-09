#!/usr/bin/env python3
"""Re-apply the `scene:` block to the task configs of the random-object tasks.

`import_tasks.py` rewrites task.yaml wholesale, which silently drops this block. Run this
after any re-import:  python apply_scene_seeds.py [--tasks ...] [--tasks-root ...]

The block pins, per task, the manipulands the frozen program NAMES, the seed to initialise
each protocol with, and the full list of seeds measured to work. Grounding is a property of
(object, scene), not of the object: a seed that spawns the right objects is not necessarily
one where perception can find them, so every listed seed was probed in that very scene
(outputs/validation/perception_fix/{noun_probe,seed_objects,usable_seeds}.py in the sim repo).
"""
import argparse, json, pathlib, sys
import yaml

DEFAULT_MEASUREMENTS = "codegen/sim/outputs/validation/perception_fix/usable_seeds.json"

# slot -> the ladder the frozen program passes, in order; and what those slots spawn
SPEC = {
    "place_a2b_left":      ({"object": ["mouse"], "target_object": ["toy car"]},
                            {"object": "mouse", "target_object": "toycar"}),
    "place_a2b_right":     ({"object": ["mouse"], "target_object": ["toy car"]},
                            {"object": "mouse", "target_object": "toycar"}),
    "place_object_basket": ({"basket": ["basket"], "object": ["playing cards", "cards"]},
                            {"basket": "basket", "object": "playingcards"}),
    "put_object_cabinet":  ({"cabinet": ["cabinet"], "object": ["playing cards"]},
                            {"cabinet": "cabinet", "object": "playingcards"}),
    "place_object_scale":  ({"scale": ["electronic scale"], "object": ["mouse"]},
                            {"object": "mouse", "scale": "electronicscale"}),
    "place_object_stand":  ({"displaystand": ["displaystand", "display stand", "stand"],
                             "object": ["rubikscube", "rubiks cube"]},
                            {"displaystand": "displaystand", "object": "rubikscube"}),
    "scan_object":         ({"scanner": ["scanner", "barcode scanner", "scan gun"],
                             "object": ["box", "tea box"]},
                            {"object": "tea-box", "scanner": "scanner"}),
}

COMMENT = ("Fixed manipulands: every listed seed spawns the SAME objects, and each seed was PROBED in "
           "that very scene -- every slot's name ladder was asked of the perception tier there and at "
           "least one name landed on the right actor. Grounding is a property of (object, scene), not of "
           "the object, so seeds are not interchangeable: replacing a seed with an unprobed one of the "
           "same objects is not supported by any measurement. NOTE: re-running import_tasks.py rewrites "
           "this file and drops this block -- re-run apply_scene_seeds.py after any re-import.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2] / "configs/robotwin/tasks_indep")
    ap.add_argument("--measurements", type=pathlib.Path, default=pathlib.Path(DEFAULT_MEASUREMENTS))
    ap.add_argument("--tasks", nargs="*", default=sorted(SPEC))
    a = ap.parse_args(argv)
    if not a.measurements.is_file():
        sys.exit(f"no measurements at {a.measurements} — the probe output is what pins these seeds")
    U = json.loads(a.measurements.read_text())
    for task in a.tasks:
        if task not in SPEC:
            sys.exit(f"{task}: not a random-object task (known: {', '.join(sorted(SPEC))})")
        p = a.tasks_root / task / "task.yaml"
        if not p.is_file():
            sys.exit(f"{task}: no {p} — run import_tasks.py first")
        ladders, models = SPEC[task]
        u, tested = U[task]["usable"], U[task]["tested"]
        meta = yaml.safe_load(p.read_text())
        meta["scene"] = {
            "comment": COMMENT,
            "models": models,
            "names": {k: v[0] for k, v in ladders.items()},
            "name_ladders": ladders,
            "seed": {"demo_clean": u["clean"][0] if u["clean"] else None,
                     "demo_randomized": u["randomized"][0] if u["randomized"] else None},
            "usable_seeds": {"demo_clean": u["clean"], "demo_randomized": u["randomized"]},
            "probe": {"demo_clean": f"{len(u['clean'])} usable of {tested['clean']} probed",
                      "demo_randomized": f"{len(u['randomized'])} usable of {tested['randomized']} probed"},
        }
        p.write_text(yaml.safe_dump(meta, sort_keys=False, width=110, allow_unicode=True))
        print(f"{task:22s} seed {meta['scene']['seed']['demo_clean']}/{meta['scene']['seed']['demo_randomized']}"
              f"   pool {len(u['clean'])}/{len(u['randomized'])}")
    print(f"{len(a.tasks)} task configs updated under {a.tasks_root}")


if __name__ == "__main__":
    main()
