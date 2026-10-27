"""Machine paths, WAM/hybrid evaluation settings, and task assets.

Evaluation settings live in configs/robotwin/eval/{wam,hybrid}.yaml.
`${key}` references site.yaml; `${repo}` is the repository root.
"""
import os
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]          # <repo>/robotwin/hybrid_eval/config.py
CFG_DIR = REPO / "configs" / "robotwin"
_REF = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")


def _lookup(scope, dotted):
    cur = scope
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"unresolved reference ${{{dotted}}}")
        cur = cur[part]
    return cur


def _resolve(obj, scope):
    if isinstance(obj, dict):
        return {k: _resolve(v, scope) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, scope) for v in obj]
    if isinstance(obj, str):
        for _ in range(10):
            new = _REF.sub(lambda m: str(_lookup(scope, m.group(1))), obj)
            if new == obj:
                return obj
            obj = new
        raise ValueError(f"reference loop in {obj!r}")
    return obj


def _load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def site_path():
    return Path(os.environ.get("AGENTICWAM_HYBRID_SITE") or CFG_DIR / "site.yaml")


def site():
    """Resolved site.yaml as a dict (with `repo` injected and env overrides applied)."""
    p = site_path()
    if not p.is_file():
        raise FileNotFoundError(f"site config not found: {p}\n"
                                f"  cp {CFG_DIR / 'site.example.yaml'} {CFG_DIR / 'site.yaml'} and edit the paths")
    raw = _load_yaml(p)
    raw["repo"] = str(REPO)
    s = _resolve(raw, raw)
    s["robotwin_root"] = os.environ.get("CAPX_ROBOTWIN_ROOT") or s["robotwin_root"]
    s["aspire_root"] = os.environ.get("AGENTICWAM_ASPIRE_ROOT") or s["aspire_root"]
    return s


def eval_config(name_or_path, site_cfg=None):
    """Load configs/robotwin/eval/<name>.yaml or an explicit YAML path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CFG_DIR / "eval" / f"{name_or_path}.yaml"
    if not p.is_file():
        known = sorted(x.stem for x in (CFG_DIR / "eval").glob("*.yaml"))
        raise FileNotFoundError(f"evaluation config not found: {p} (known: {', '.join(known)})")
    raw = _load_yaml(p)
    # Site keys win: an evaluation key may share the name of a site reference
    # (`dataset_stats: ${dataset_stats}`), which would otherwise resolve to itself
    scope = dict(raw)
    scope.update(site_cfg or site())
    m = _resolve(raw, scope)
    m["name"] = p.stem
    for key in ("checkpoint", "task_assets_dir"):
        if m.get(key):
            p = Path(m[key]).expanduser()
            m[key] = str(p if p.is_absolute() else REPO / p)
    return m


def resolve_checkpoint(model_cfg, checkpoint=None):
    """Use an explicit checkpoint or the configured WAM/hybrid weights."""
    if checkpoint:
        p = Path(checkpoint).expanduser()
    elif model_cfg.get("checkpoint"):
        p = Path(model_cfg["checkpoint"])
    else:
        raise ValueError("give --ckpt or set checkpoint in the evaluation config")
    if not p.is_file():
        raise FileNotFoundError(
            f"no checkpoint: {p}\nDownload the release with:\n"
            "  hf download miniFranka/agentic-wam-posttrain-robotwin-suit "
            "--local-dir checkpoints/robotwin_release\n"
            "or pass --ckpt and configure dataset_stats in site.yaml."
        )
    return str(p.resolve())


def tasks_root(model_cfg=None):
    return Path(os.environ.get("AGENTICWAM_HYBRID_TASKS")
                or (model_cfg or {}).get("task_assets_dir") or CFG_DIR / "tasks")


def list_tasks(model_cfg=None):
    return sorted(d.name for d in tasks_root(model_cfg).iterdir() if (d / "frozen_program.py").is_file())


def load_task_yaml(name):
    p = tasks_root() / name / "task.yaml"
    return _load_yaml(p) if p.is_file() else {}


def task(name, model_cfg=None):
    """Assets + per-task defaults for one task directory. Raises if the directory or a file is missing."""
    root = tasks_root(model_cfg)
    d = root / name
    if not d.is_dir():
        raise FileNotFoundError(f"task {name!r} has no directory under {root} (known: {', '.join(list_tasks(model_cfg))})")
    t = {"name": name, "dir": str(d), "root": str(root),
         "preamble": str(d / "preamble.txt"), "frozen": str(d / "frozen_program.py")}
    for k in ("preamble", "frozen"):
        if not Path(t[k]).is_file():
            raise FileNotFoundError(f"task {name}: missing {t[k]}")
    meta = _load_yaml(d / "task.yaml") if (d / "task.yaml").is_file() else {}
    t["short"] = meta.get("short", name)
    for k in ("gen_tokens", "nis"):
        t[k] = meta.get(k, (model_cfg or {}).get(k))
    return t
