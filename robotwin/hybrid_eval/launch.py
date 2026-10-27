#!/usr/bin/env python3
"""One RoboTwin eval arm: hybrid (generated code x WAM, AGENTICWAM_HYBRID=1) or pure WAM.

    python robotwin/hybrid_eval/launch.py --task beat_block_hammer --model robotwin_hybrid --step 15000 \
        --seed 0 --episodes 5 --shard-gpu 6 --vis-gpu 7 [--proto demo_randomized] [--mode wam] [--dry-run]

Everything machine-specific comes from configs/robotwin/site.yaml, the checkpoint family from
configs/robotwin/models/<model>.yaml, the task from configs/robotwin/tasks/<task>/ (preamble.txt,
frozen_program.py, task.yaml). The arm writes run_meta.json beside its results so rollout_viewer.py needs no
naming conventions or hand-written baseline maps: a hybrid run pairs with the WAM run that has the same
(model, step, task, protocol, seed).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_eval import config, vision  # noqa: E402

REPO = config.REPO
EVAL_ENTRY = REPO / "robotwin" / "eval" / "run_task.py"


def ckpt_tag(ckpt):
    """Mirror of robotwin/eval/run_task.py:_resolve_ckpt_tag: results go under evaluate_results/robotwin/<tag>/."""
    parts = Path(ckpt).resolve().parts
    if "runs" in parts:
        i = parts.index("runs")
        return f"{parts[i + 1]}_{parts[i + 2]}"
    return Path(ckpt).stem


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="a directory name under configs/robotwin/tasks/")
    ap.add_argument("--model", default="robotwin_hybrid", help="preset name under configs/robotwin/models/ or a yaml path")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step under <run>/checkpoints/weights/")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (default: preset's released checkpoint)")
    ap.add_argument("--mode", choices=["hybrid", "wam"], default="hybrid")
    ap.add_argument("--proto", default="demo_clean", help="RoboTwin task_config: demo_clean | demo_randomized")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--shard-gpu", type=int, required=True, help="GPU for the policy")
    ap.add_argument("--vis-gpu", type=int, default=None, help="GPU for Molmo (hybrid mode)")
    ap.add_argument("--vis-gpu-seg", type=int, default=None, help="GPU for SAM3 + GraspNet (default: --vis-gpu)")
    ap.add_argument("--molmo-util", type=float, default=None)
    ap.add_argument("--tag", default=None, help="run tag (default <mode>_<short>_<proto>_s<step>_seed<seed>)")
    ap.add_argument("--baseline", default=None, help="tag of the pure-WAM run to compare against (default: auto by meta)")
    ap.add_argument("--gen-tokens", type=int, default=None)
    ap.add_argument("--nis", type=int, default=None)
    ap.add_argument("--replan", type=int, default=None)
    ap.add_argument("--frame-every", type=int, default=15, help="AGENTICWAM_FRAME_EVERY: one video frame per N sim steps")
    ap.add_argument("--perc-version", default="r10", help="CAPX_PERC_VERSION")
    # Models trained on the RANDOMISED `seen` instruction pool (record_instructions=static, ~3 per
    # episode rotating by task_index) have no canonical primary, so no canonical rewrite is offered here. Evals default to `unseen`: this knob exists to measure that gap.
    ap.add_argument("--instruction-type", choices=["seen", "unseen"], default=None,
                    help="EVALUATION.instruction_type; default uses the preset, then the simulator config")
    ap.add_argument("--no-vision", action="store_true", help="hybrid mode: assume the vision servers are up")
    ap.add_argument("--no-render", action="store_true",
                    help="disable rollout video writing (EVALUATION.eval_video_log=false)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra hydra overrides appended verbatim, e.g. --extra EVALUATION.foo=1")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved plan and the command, start nothing")
    return ap.parse_args(argv)


CAPDATA = "/path/to/robotwin_cap_slices"   # fallback slice corpus; site.yaml robotwin_cap_slices / env win



def nn_repair_corpus(eval_task):
    """Root of the GT slice corpus for a preset, derived from the task config it already names.

    NN-repair must match against the corpus the checkpoint TRAINED on. Pointing it elsewhere does
    not fail: the stage split and helper names differ, every similarity lands near 0.2 against a
    0.80 threshold, and the feature is silently inert (161 episodes, zero repairs, when a
    fixed default pointed at another corpus). Deriving it from `eval_task` -- the same key that already fixes the canvas
    and token budget -- removes the chance to disagree. Chain:

        preset.eval_task -> configs/robotwin/task/<eval_task>.yaml `override /data: <name>`
                         -> configs/robotwin/data/<name>.yaml `code_slices_cache_dir:` (one per task)
                         -> strip the trailing `<task>/raw/slices/slices_ep` -> the shared root
    """
    tcfg = REPO / "configs" / "robotwin" / "task" / f"{eval_task}.yaml"
    data_name = eval_task
    if tcfg.is_file():
        m = re.search(r"override\s+/data:\s*([A-Za-z0-9_]+)", tcfg.read_text(encoding="utf-8"))
        if m:
            data_name = m.group(1)
    dcfg = REPO / "configs" / "robotwin" / "data" / f"{data_name}.yaml"
    if not dcfg.is_file():
        raise RuntimeError(f"cannot derive the NN-repair corpus: {dcfg} not found")
    roots = {re.sub(r"/[^/]+/raw/slices/slices_ep/?$", "", d)
             for d in re.findall(r"code_slices_cache_dir:\s*(\S+)", dcfg.read_text(encoding="utf-8"))}
    if len(roots) != 1:
        raise RuntimeError(f"{dcfg.name}: expected exactly one slice-corpus root, got {sorted(roots)}")
    return roots.pop()


def measured_L(task, code_model):
    """Canvas = stage-1 GT tokens + 1 EOS, tokenized NOW with the code model's tokenizer.

    tasks/<t>/task.yaml records a `tokens:` block, but it drifts: three of ten tasks checked on
    2026-09-13 were stale, in both directions (blocks_ranking_size 88 vs 93, place_cans_plasticbox
    118 vs 121, place_burger_fries 135 vs 129). A canvas with surplus is the worse failure -- the
    model fills it by padding the program into unparseable text, which the gate then rejects
    (observed: 1/37 gate passes at canvas 87 vs 21/21 at the correct 82). Returns None if the
    slice or tokenizer is unavailable, so the caller falls back to the old behaviour.
    """
    try:
        from transformers import AutoTokenizer
        p = Path(CAPDATA) / task / "raw" / "slices" / "slices_ep" / "ep_000000.json"
        if not p.is_file():
            return None
        slices = json.loads(p.read_text())["slices"]
        tk = AutoTokenizer.from_pretrained(code_model)
        return len(tk.encode(slices[0], add_special_tokens=False)) + 1
    except Exception:
        return None


def plan(a):
    site = config.site()
    model = config.model_preset(a.model, site)
    # `task_assets: false` in a preset = a WAM-only checkpoint whose code prompt does not come from the
    # task's preamble (e.g. the RoboTwin 2.0 benchmark runs, trained on skill-prompt caches): no task
    # directory is needed and no preamble / canvas / history env is exported.
    assets = bool(model.get("task_assets", True))
    if not assets and a.mode == "hybrid":
        sys.exit(f"preset {model['name']} has task_assets: false; the hybrid mode needs the task's frozen program")
    task = config.task(a.task, model) if assets else {"name": a.task, "short": a.task, "preamble": None,
                                                      "frozen": None, "gen_tokens": None}
    use_code = bool(model.get("use_code_expert", True))
    try:
        ckpt = config.resolve_checkpoint(model, a.ckpt, a.step)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    step = a.step if a.step is not None else (int(m.group(1)) if (m := re.search(r"step_(\d+)", ckpt)) else None)
    proto_short = a.proto.replace("demo_", "")
    tag = a.tag or f"{a.mode}_{task['short']}_{proto_short}_s{step}_seed{a.seed}"
    if assets:
        # A fixed training canvas is a checkpoint property. Per-task lengths must
        # not shrink the hybrid checkpoint's 124 slots (click_alarmclock's metadata says 72).
        gen_tokens = a.gen_tokens or model.get("gen_tokens") \
            or measured_L(a.task, model["code_model"]) or task["gen_tokens"]
    else:
        gen_tokens = a.gen_tokens or model.get("gen_tokens")
    # task["nis"] is DEPRECATED and deliberately not consulted. import_tasks.py derives it as
    # gen_tokens * 40/96, i.e. it scales with the CODE canvas -- but nis lands on
    # EVALUATION.num_inference_steps, which only sets video_scheduler.set_timesteps(), the
    # video/action denoiser. How many denoising steps the video needs has nothing to do with how
    # long the code slice is, so the per-task value varied an unrelated knob across tasks
    # (31..53 on the current corpus). Every run so far passed --nis explicitly and never hit it.
    nis = a.nis or model["nis"]
    replan = a.replan or model.get("replan", 24)
    instruction_type = a.instruction_type or model.get("instruction_type")
    results_dir = REPO / "evaluate_results" / "robotwin" / ckpt_tag(ckpt) / f"{tag}_{a.task}_g0_s{a.seed}" / a.task
    env = dict(os.environ)
    env.update({
        "TASK": a.task,
        "PATH": f"{site['python_bin']}:{env.get('PATH', '')}",
        "PYTHONPATH": f"{REPO / 'src'}:{env.get('PYTHONPATH', '')}",
        "VK_ICD_FILENAMES": site["vulkan_icd"],
        "LD_LIBRARY_PATH": ":".join(site.get("ld_library_paths", []) + [env.get("LD_LIBRARY_PATH", "")]),
        "DIFFSYNTH_DOWNLOAD_SOURCE": "huggingface", "DIFFSYNTH_MODEL_BASE_PATH": site["diffsynth_model_base"],
        "TORCHDYNAMO_DISABLE": "1", "TORCH_COMPILE_DISABLE": "1", "HYDRA_FULL_ERROR": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "AGENTICWAM_ACTION_SPACE": model["action_space"],
        "AGENTICWAM_CUDA_VISIBLE_DEVICES": str(a.shard_gpu),
        "HF_HOME": site["hf_cache_policy"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    if assets:
        env.update({"AGENTICWAM_CODE_PREAMBLE_FILE": task["preamble"],
                    "AGENTICWAM_HYBRID_TASKS": task["root"],
                    "AGENTICWAM_CODE_NO_HISTORY": str(model.get("no_history", 0))})
    else:
        env.pop("AGENTICWAM_CODE_PREAMBLE_FILE", None)
        env.pop("AGENTICWAM_CODE_NO_HISTORY", None)
    if gen_tokens:
        env["AGENTICWAM_INFER_CODE_TOKENS"] = str(gen_tokens)
    # preset-level environment (instruction maps, skill prompts, decode knobs)
    env.update({k: str(v) for k, v in (model.get("env") or {}).items()})
    env["AGENTICWAM_HYBRID"] = "1" if a.mode == "hybrid" else "0"
    if a.mode == "hybrid":
        env.update({
            "AGENTICWAM_HYBRID": "1", "AGENTICWAM_HYBRID_PERCEPTION": "1", "CAPX_USE_SAM3": "1",
            # Derived, not configured -- see nn_repair_corpus. An explicit AGENTICWAM_NN_REPAIR_CORPUS in
            # the environment still wins (setdefault below), for debugging against another corpus.
            # Resolve from __file__ rather than REPO/robotwin so the arbiter that
            # runs is always the one sitting next to this launcher. That mattered when the
            # fixed tree lived alongside the original as robotwin_v2 (2026-09-15, to keep an
            # in-flight run on its original code); the copy was folded back on 2026-09-16 and
            # the original archived as robotwin_pre_v2_20260916. Keep it derived:
            # it is what lets a tree be copied aside and tested without editing any caller.
            "AGENTICWAM_HYBRID_PKG": str(Path(__file__).resolve().parents[1]),
            "AGENTICWAM_FROZEN_PROGRAM": task["frozen"],
            "CAPX_PERC_VERSION": a.perc_version, "AGENTICWAM_FRAME_EVERY": str(a.frame_every),
            "CAPX_ROBOTWIN_ROOT": site["robotwin_root"], "AGENTICWAM_ASPIRE_ROOT": site["aspire_root"],
            "AGENTICWAM_TRACE": f"{site['trace_dir']}/trace_{tag}",
        })
        # Only when NN-repair is on: a run without it needs no corpus, and a data config that
        # carries no slice caches (a no-slice ablation) would otherwise fail the derivation for
        # no reason. setdefault, so an explicit export still wins.
        if env.get("AGENTICWAM_NN_REPAIR", "0") == "1":
            env.setdefault("AGENTICWAM_NN_REPAIR_CORPUS", nn_repair_corpus(model["eval_task"]))
            print("nn-repair corpus:", env["AGENTICWAM_NN_REPAIR_CORPUS"])
    overrides = [
        f"ckpt={ckpt}", "gpu_id=0", f"seed={a.seed}",
        f"EVALUATION.robotwin_root={site['robotwin_root']}",
        f"EVALUATION.task_name={a.task}", f"EVALUATION.task_config={a.proto}",
        f"EVALUATION.eval_num_episodes={a.episodes}", f"task={model['eval_task']}",
        f"EVALUATION.num_inference_steps={nis}", f"EVALUATION.replan_steps={replan}",
        *([f"EVALUATION.instruction_type={instruction_type}"] if instruction_type else []),
        f"EVALUATION.output_dir=eval_shards/{tag}_{a.task}_g0_s{a.seed}",
        f"model.use_code_expert={'true' if use_code else 'false'}",
        *([f"model.code_model_name={model['code_model']}"] if use_code else []),
        # run_task.py auto-finds dataset_stats.json only in the ckpt's parent dirs,
        # which works for runs/<name>/checkpoints/weights/ but NOT for a checkpoint copied out
        # of the run (e.g. a checkpoint snapshot directory). Pass it explicitly when site.yaml
        # names it, so --ckpt works from anywhere.
        *([f"EVALUATION.dataset_stats_path={model.get('dataset_stats') or site['dataset_stats']}"]
          if (model.get("dataset_stats") or site.get("dataset_stats")) else []),
        f"model.action_dim={model['action_dim']}", f"model.proprio_dim={model['proprio_dim']}",
        *(["EVALUATION.eval_video_log=false"] if a.no_render else []),
        *a.extra,
    ]
    cmd = [f"{site['python_bin']}/python", str(EVAL_ENTRY)] + overrides
    meta = {"env": "robotwin_sim", "tag": tag, "mode": a.mode, "task": a.task, "task_short": task["short"], "proto": a.proto,
            "seed": a.seed, "model": model["name"], "step": step, "ckpt": str(ckpt), "episodes": a.episodes,
            "gen_tokens": gen_tokens, "nis": nis, "replan": replan, "baseline": a.baseline,
            "use_code_expert": use_code, "eval_task": model["eval_task"],
            "instruction_type": instruction_type,
            "eval_preproc": env.get("AGENTICWAM_EVAL_PREPROC", "parity"),
            "eval_preproc_crop": float(env.get("AGENTICWAM_EVAL_PREPROC_CROP", "0.95")),
            "decode": {"conf_thresh": float(env.get("AGENTICWAM_CONF_THRESH", "0")),
                       "reveal_cap": int(env.get("AGENTICWAM_REVEAL_CAP", "9999")),
                       "nn_repair": env.get("AGENTICWAM_NN_REPAIR", "0") == "1"},
            "hybrid": {"perception": True, "perc_version": a.perc_version, "frame_every": a.frame_every,
                       "preamble": task["preamble"], "frozen": task["frozen"]} if a.mode == "hybrid" else None,
            "host": os.uname().nodename, "shard_gpu": a.shard_gpu, "started": None, "finished": None, "result": None}
    return {"site": site, "model": model, "task": task, "tag": tag, "results_dir": results_dir,
            "env": env, "cmd": cmd, "meta": meta,
            "log": REPO / "runs" / "eval_logs" / f"{tag}_{a.task}_s{a.seed}.log"}


def write_meta(p):
    p["results_dir"].mkdir(parents=True, exist_ok=True)
    (p["results_dir"] / "run_meta.json").write_text(json.dumps(p["meta"], indent=1))


def main(argv=None):
    a = parse_args(argv)
    p = plan(a)
    site, meta = p["site"], p["meta"]
    watch = Path(site["watch_log"]); watch.parent.mkdir(parents=True, exist_ok=True)

    def say(msg):
        line = f"[{datetime.now():%F %T}] {msg}"
        print(line, flush=True)
        with open(watch, "a") as fh:
            fh.write(line + "\n")

    if a.dry_run:
        print(json.dumps({k: meta[k] for k in ("tag", "mode", "task", "proto", "seed", "model", "step", "ckpt",
                                                "episodes", "gen_tokens", "nis", "replan")}, indent=1))
        print("results:", p["results_dir"])
        print("env (AGENTICWAM_/CAPX_):", {k: v for k, v in p["env"].items() if k.startswith(("AGENTICWAM_", "CAPX_"))})
        print("cmd:", " ".join(p["cmd"]))
        return 0
    if a.mode == "hybrid" and not a.no_vision:
        if a.vis_gpu is None:
            sys.exit("hybrid mode needs --vis-gpu (or --no-vision if the servers are already up)")
        if not vision.ensure_up(site, a.vis_gpu, a.vis_gpu_seg, a.molmo_util, say):
            return 1
    Path(site["trace_dir"]).mkdir(parents=True, exist_ok=True)
    meta["started"] = datetime.now().isoformat(timespec="seconds")
    write_meta(p)
    say(f"=== {meta['mode'].upper()} armed: {meta['tag']} task={meta['task']} proto={meta['proto']} step={meta['step']} "
        f"seed={meta['seed']} N={meta['episodes']} canvas={meta['gen_tokens']} NIS={meta['nis']} shard GPU {a.shard_gpu} ===")
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(p["log"], "a") as fh:                 # append: a retried job keeps the earlier attempt's error
        start = fh.tell()
        fh.write(f"\n===== {meta['started']} attempt on {meta['host']} GPU {a.shard_gpu} =====\n"); fh.flush()
        rc = subprocess.run(p["cmd"], cwd=REPO, env=p["env"], stdout=fh, stderr=subprocess.STDOUT).returncode
    with open(p["log"], errors="replace") as fh:
        fh.seek(start)
        txt = re.sub(r"\x1b\[[0-9;]*m", "", fh.read())   # this attempt only
    hits = re.findall(r"Success rate: (\d+)/(\d+) => ([0-9.]+)%", txt)
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    meta["wall_s"] = round(time.time() - t0)
    meta["result"] = {"success": int(hits[-1][0]), "total": int(hits[-1][1]), "rc": rc} if hits else {"rc": rc}
    write_meta(p)
    say(f"{meta['tag']} done: " + (f"{hits[-1][0]}/{hits[-1][1]} => {hits[-1][2]}%" if hits else f"NO RESULT rc={rc} (see {p['log']})"))
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
