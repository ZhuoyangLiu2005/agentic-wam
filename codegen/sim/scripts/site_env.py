#!/usr/bin/env python3
"""site.yaml -> shell exports. `env.sh` evals this; every script and module then reads plain environment variables.

    python3 scripts/site_env.py [--site <yaml>] [--check]

Prints `export NAME='value'` for each site key whose variable is not already set (the caller wins); `--check`
exits non-zero listing variables still unset. Stdlib + PyYAML only.
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# site key (dotted for nested) -> environment variable
KEYS = {
    "robotwin_py": "ROBOTWIN_PY",
    "robotwin_root": "CAPX_ROBOTWIN_ROOT",
    "trainer_root": "TRAINER_ROOT",
    "train_py": "TRAIN_PY",
    "slicer": "SLICER",
    "data_root": "CODEGEN_SIM_DATA_ROOT",
    "agent_root": "CODEGEN_SIM_AGENT_ROOT",
    "validation_root": "CODEGEN_SIM_VALIDATION_ROOT",
    "log_root": "CODEGEN_SIM_LOG_ROOT",
    "rollout_cache": "CAPX_ROLLOUT_CACHE",
    "vision.molmo": "MOLMO_SERVICE_URL",
    "vision.sam3": "SAM3_SERVICE_URL",
    "vision.graspnet": "GRASPNET_SERVICE_URL",
    "gpu.min_free_mib": "CODEGEN_SIM_GPU_MIN_FREE_MIB",
    "gpu.replays_per_card": "CODEGEN_SIM_REPLAYS_PER_CARD",
    "trial_timeout_s": "CAPSIM_TRIAL_TIMEOUT_S",
    "ffmpeg_dir": "CODEGEN_SIM_FFMPEG_DIR",
}
REQUIRED = ("ROBOTWIN_PY", "CAPX_ROBOTWIN_ROOT")      # TRAINER_ROOT / TRAIN_PY are checked by make_dataset.py's export step
DEFAULTS = {"robotwin_root": "${repo}/capsim/third_party/RoboTwin",
            "slicer": "${trainer_root}/robotwin/dataprep/slicing.py"}
_REF = re.compile(r"\$\{([a-z_.]+)\}")


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = "" if v is None else str(v)
    return out


def resolve(flat):
    flat = dict(flat)
    flat.setdefault("repo", str(ROOT))
    flat.setdefault("user", os.environ.get("USER") or os.environ.get("LOGNAME") or "user")
    for _ in range(10):
        changed = False
        for k, v in flat.items():
            new = _REF.sub(lambda m: flat.get(m.group(1), m.group(0)), v)
            if new != v:
                flat[k] = new; changed = True
        if not changed:
            break
    for k, v in flat.items():
        if _REF.search(v):
            raise ValueError(f"site.yaml: unresolved reference in {k}: {v}")
    return flat


def load(site_path):
    import yaml
    flat = _flatten(yaml.safe_load(site_path.read_text()) or {}) if site_path.is_file() else {}
    if os.environ.get("TRAINER_ROOT"):
        flat.setdefault("trainer_root", os.environ["TRAINER_ROOT"])
    for key, val in DEFAULTS.items():
        if key not in flat and all(r in flat or r in ("repo", "user") for r in _REF.findall(val)):
            flat[key] = val
    return resolve(flat), site_path.is_file()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.environ.get("CODEGEN_SIM_SITE") or str(ROOT / "site.yaml"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    site_path = Path(a.site)
    flat, found = load(site_path)
    exports = {}
    if flat:
        for key, var in KEYS.items():
            if os.environ.get(var):
                continue
            val = flat.get(key, "")
            if not val:
                continue
            if key == "robotwin_root":
                val = os.path.realpath(val)
            exports[var] = val
    if a.check:
        missing = [v for v in REQUIRED if not (os.environ.get(v) or exports.get(v))]
        if not found:
            print(f"env.sh: no site file at {site_path} (cp site.example.yaml site.yaml and edit), and the environment "
                  f"lacks: {', '.join(missing) or 'nothing'}", file=sys.stderr)
        elif missing:
            print(f"env.sh: {site_path} does not provide: {', '.join(missing)}", file=sys.stderr)
        return 1 if missing else 0
    for var, val in exports.items():
        print(f"export {var}='{val}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
