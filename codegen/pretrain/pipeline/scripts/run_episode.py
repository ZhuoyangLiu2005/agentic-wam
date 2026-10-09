#!/usr/bin/env python
"""VLM-in-the-loop INVERSE code generation for logged YAM bimanual data — CLI.

A forward agent perceives a scene and writes API code to act; here the trajectory
already EXISTS, so per SLICE we recover the code that EXPLAINS it. The program is written
against the ACTIVE embodiment's declaration-only API (YAM:
`codegen.api.bimanual.YamBimanualApi`; ego: `codegen.api.ego.EgoHandApi` —
nothing executes), reflected into the
prompt via `combined_doc()` exactly as the forward workflow builds its API block.

Per episode:
  1. SLICE the trajectory to TILE [0,T): one slice per annotated sub_task block, plus one
     per coalesced UNLABELED span.
  2. GENERATE slice by slice, CONTINUING one program: ONE VLM call per slice — the model
     gets the PROGRAM SO FAR (variables persist), the slice's frames, and the observed
     gripper events, reads the frames itself, and emits ONLY the new lines
     (there is no separate perceive/describe stage).
  3. VALIDATE each slice statically (codegen.program.validate): it must PARSE in program
     context and only call real API functions with acceptable arguments;
     failures go back to the same model for up to --repair-retries rewrites, and a slice
     that still doesn't parse is committed commented-out so program.py always compiles.
  4. GROUND to the gripper sensor: the observed open/close transitions are matched to those
     IMPLIED by the code (codegen.program.effects) -- every API fn (primitive
     `open/close_gripper` OR semantic `pick`/`place`/...) has a declared gripper CONTRACT
     (the embodiment's *_contracts table). On a match, each transition
     frame is pinned to its producing line (grip_pins); otherwise frames map to code at the
     sub-task BLOCK level.

Models run via the configured OpenAI-compatible endpoint (VLM_URL; key in VLM_API_KEY or .vlm_api_key); default = Gemini 3.1 Pro,
and --coder-model accepts any family it hosts (Gemini / Claude / GPT).
Run from the workspace root (PY=.venv/bin/python; needs the top-cam clip from
the raw LeRobot dataset):

    $PY scripts/run_episode.py \
        --task load_batteries_into_a_calculator [--max-slices N]

Outputs (schema: codegen/pipeline/artifacts.py): program.py, lines.json
(frame_src_block + grip_pins), chunks.json, trace.json (per-slice I/O),
trace/*.jpg — in the run dir given by the embodiment spec.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen.actor.client import MODEL           # noqa: E402  (vocabulary-free)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    from codegen import config
    ap.add_argument("--embodiment", default=config.embodiment(),
                    choices=["yam", "ego", "tianji_sharpa", "yam_sft"],
                    help="which embodiment's data/API/library this actor uses "
                         "(default: CODEGEN_EMBODIMENT env / the config file)")
    ap.add_argument("--task", required=True, help="dataset dir under the source root (e.g. train_shard_0)")
    ap.add_argument("--coder-model", default=MODEL,
                    help="the VLM that reads the frames and writes the code (one call per slice)")
    ap.add_argument("--frame-gap", type=int, default=0,
                    help="max frames between consecutive VLM frames of a slice: "
                         "endpoints + event pre-frames anchor the sample, and any "
                         "longer interval is filled with evenly spaced frames "
                         "(no events -> uniform sampling). Default 0 = the active "
                         "embodiment's config value (YAM_FRAME_GAP / EGO_FRAME_GAP)")
    ap.add_argument("--vanilla", action="store_true", default=None,
                    help="annotation-only prompt AND uniform sampling (full "
                         "detector-influence ablation); reconciliation still "
                         "runs afterwards as an independent eval. Default: the "
                         "`vanilla` config key (flag present = force on)")
    ap.add_argument("--max-slices", type=int, default=config.get_int("MAX_SLICES"),
                    help="0 = all (smoke-test with a small N)")
    ap.add_argument("--repair-retries", type=int, default=config.get_int("REPAIR_RETRIES"),
                    help="max repair rounds per slice when static validation fails (0 = report only)")
    ap.add_argument("--episode", type=int, default=0,
                    help="episode index within the task's dataset (multi-episode "
                         "corpora; run dirs are per-episode via episode_id)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--tag", default=config.get("TAG"),
                    help="campaign version tag for the output dir — separates rounds "
                         "so runs with a grown skill library never overwrite the "
                         "baseline. Default: the `tag` config key")
    args = ap.parse_args()
    if args.vanilla is None:
        args.vanilla = config.get_bool("VANILLA")
    # The active-embodiment surface resolves from this env var at codegen import
    # time, so it must be set BEFORE the pipeline import below.
    os.environ["CODEGEN_EMBODIMENT"] = args.embodiment
    if not args.frame_gap:      # per-embodiment default from config
        args.frame_gap = config.frame_gap(args.embodiment)
    from codegen.pipeline.episode import run_episode  # noqa: PLC0415
    return run_episode(args)


if __name__ == "__main__":
    raise SystemExit(main())
