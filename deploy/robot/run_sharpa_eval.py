#!/usr/bin/env python3
"""Run AgenticWAM policy evaluation on the vega + Sharpa robot -- the twin of run_agenticwam_eval.py (YAM).

    # what the model will see -- NOTHING MOVES
    python deploy/robot/run_sharpa_eval.py --instruction "<task>" --preview-only
    # full loop: inference + guard, but nothing is sent to the robot (no reset move either)
    python deploy/robot/run_sharpa_eval.py --instruction "<task>" --dry-run
    # real run
    python deploy/robot/run_sharpa_eval.py --instruction "<task>"

Run it in the T-Rex hardware venv (Python 3.10, deploy/robot/vega/README.md lists the packages) on the robot
station, with `bash deploy/launch_server_sharpa.sh` serving on the same machine (127.0.0.1:8712; the token is read
from runs/deploy_sharpa/token.txt). --preview-only also checks the camera image size: the client refuses frames that
are not 640x360, and the server must refuse a 320x224 head frame.

Control loop is sequential, like YAM and T-Rex eval: capture -> POST /act (the arms hold their last target
while the model thinks) -> guard the chunk -> execute `replan_steps` rows at 30 Hz -> repeat. Row 0 of each
chunk therefore corresponds to the measured state. Episode end (s/f/q + Enter, max replans, a guard reject, an
inference error, or the 300 Hz thread's own safety stop) holds the arms; reset by hand or with the next
episode's collision-planned move to the start pose.

Each replan's generated program is printed (repeats collapse to one line) and saved under
runs/sharpa_eval/<session>/ep_NNN/code/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import select
import sys
import time

from loguru import logger
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # deploy/ -> common.sharpa_obs
import sharpa_client  # noqa: E402
import sharpa_guard  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _poll_stdin() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        return line.strip().lower() if line else None
    return None


def preview_grid(images: dict) -> np.ndarray:
    """Top: raw 640x360 frames. Bottom: the exact 320x224 model view (deploy/common/sharpa_obs.py). BGR."""
    import cv2
    from common import sharpa_obs

    raw = [cv2.resize(images[k], (320, 180)) for k in sharpa_client.CAMERAS]
    model = [sharpa_obs.head_camera_view(images["head_camera"]),
             sharpa_obs.wrist_camera_view(images["left_camera"]),
             sharpa_obs.wrist_camera_view(images["right_camera"])]
    grid = np.vstack([np.hstack(raw), np.hstack(model)])
    return cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)


class EpisodeRecorder:
    """ep_NNN/{meta.json, code/replan_NNN.txt + .timing.json, obs/replan_NNN.jpg, steps.jsonl, result.json}."""

    def __init__(self, root: Path, episode: int, meta: dict, save_obs: bool):
        self.dir = root / f"ep_{episode:03d}"
        (self.dir / "code").mkdir(parents=True, exist_ok=True)
        self.save_obs = save_obs
        if save_obs:
            (self.dir / "obs").mkdir(exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._steps = (self.dir / "steps.jsonl").open("w")

    def replan(self, index: int, code: str | None, images: dict, info: dict) -> None:
        if code:
            (self.dir / "code" / f"replan_{index:03d}.txt").write_text(code)
        (self.dir / "code" / f"replan_{index:03d}.timing.json").write_text(
            json.dumps({k: v for k, v in info.items() if k != "code"}, indent=2))
        if self.save_obs:
            import cv2
            cv2.imwrite(str(self.dir / "obs" / f"replan_{index:03d}.jpg"), preview_grid(images))

    def step(self, **row) -> None:
        self._steps.write(json.dumps(row) + "\n")

    def finish(self, result: dict) -> Path:
        self._steps.close()
        (self.dir / "result.json").write_text(json.dumps(result, indent=2))
        return self.dir


def preflight(health: dict, obs: dict, preview_path: Path) -> bool:
    """Print everything the model will consume; the operator confirms before anything moves."""
    print("\n" + "=" * 70)
    print(f"  server        task={health.get('task')} embodiment={health.get('embodiment')} "
          f"ckpt={health.get('ckpt')}")
    print(f"  model         horizon={health.get('action_horizon')} replan={health.get('replan_steps')} "
          f"nis={health.get('num_inference_steps')} code_expert={health.get('code_expert')}")
    ok = True
    for key in sharpa_client.CAMERAS:
        img, age = obs["images"][key], obs["frame_age_s"][key]
        good = img.shape == sharpa_client.RAW_SHAPE
        ok &= good
        print(f"  {key:13s} {img.shape[1]}x{img.shape[0]} age {age * 1e3:5.0f} ms {'OK' if good else 'WRONG SIZE'}")
    print(f"  state         58-D, arms L {np.round(obs['state'][0:7], 2)} R {np.round(obs['state'][29:36], 2)}")
    print(f"  preview       {preview_path}  (top: raw, bottom: model view)")
    print("=" * 70)
    if health.get("smoke_mode"):
        print("  SERVER IS IN SMOKE MODE (no checkpoint) -- refusing to run.")
        return False
    return ok


def run_episode(a, robot, client, guard, recorder, instruction: str) -> dict:
    stop_reason, step_index, prev_code, n_replans = "max_replans", 0, None, 0
    start = time.monotonic()
    robot.reset_clock()
    for replan in range(a.max_replans):
        if not robot.command_thread_alive:
            stop_reason = "command_thread_stopped"   # the 300 Hz thread's tracking-error safety stop
            break
        obs = robot.get_observation()
        try:
            chunk, info = client.infer(obs, task=instruction)
        except sharpa_client.AgenticWamServerError as exc:
            logger.error("Inference failed: {}", exc)
            stop_reason = "inference_failed"
            break
        n_replans += 1
        sharpa_client.print_code(replan, info["code"], prev_code)
        prev_code = info["code"]
        recorder.replan(replan, info["code"], obs["images"], info)

        rows = min(info["replan_steps"], len(chunk))
        reason = guard.check_chunk(chunk, obs["state"], rows)
        logger.info("replan {:3d} | rtt {:5.0f} ms (model {:4.0f}) | rows {}{}", replan, info["round_trip_ms"],
                    info["server_infer_ms"], rows, f" | REJECTED: {reason}" if reason else "")
        if reason:
            stop_reason = "guard_reject"
            break

        for row in range(rows):
            safe, clipped = guard.filter_row(chunk[row])
            if not a.dry_run:
                robot.command(safe)
            recorder.step(step=step_index, replan=replan, row=row, t_ns=time.monotonic_ns(),
                          state=robot.get_state().tolist(), action_raw=chunk[row].tolist(),
                          action_sent=None if a.dry_run else safe.tolist(), clipped=clipped,
                          round_trip_ms=info["round_trip_ms"] if row == 0 else None)
            step_index += 1
            robot.wait_for_next_step()

        command = _poll_stdin()
        if command in {"s", "f", "q"}:
            stop_reason = {"s": "success", "f": "failure", "q": "aborted"}[command]
            logger.info("Operator: {}", stop_reason)
            break
    robot.hold()

    verdict = stop_reason if stop_reason in {"success", "failure"} else None
    if verdict is None and stop_reason != "aborted":
        answer = input("\nOutcome? [s]uccess / [f]ailure / [Enter] skip: ").strip().lower()
        verdict = {"s": "success", "f": "failure"}.get(answer)
    return {"stop_reason": stop_reason, "verdict": verdict, "replans": n_replans, "steps": step_index,
            "duration_s": time.monotonic() - start, "dry_run": a.dry_run, "latency": client.stats.summary(),
            "guard": guard.summary(), "control": {"overrun_rate": robot.overrun_rate}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instruction", required=True, help="the dataset's task string (as in training)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--token-file", type=Path, default=REPO / "runs/deploy_sharpa/token.txt",
                    help="default: the token launch_server_sharpa.sh wrote on this machine; $AGENTICWAM_DEPLOY_TOKEN wins")
    ap.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-replans", type=int, default=60,
                    help="60 x 24 rows = 48 s at 30 Hz; demo episodes run ~45 s")
    ap.add_argument("--preview-only", action="store_true", help="capture + preview, nothing moves")
    ap.add_argument("--dry-run", action="store_true", help="full loop, nothing is sent to the robot")
    ap.add_argument("--no-save-obs", action="store_true", help="do not save the per-replan model-view jpg")
    ap.add_argument("--row0-max", type=float, default=sharpa_guard.GuardLimits.row0_max)
    ap.add_argument("--step-count", type=float, default=sharpa_guard.GuardLimits.step_count)
    ap.add_argument("--step-reject", type=float, default=sharpa_guard.GuardLimits.step_reject)
    ap.add_argument("--robot-config", type=Path, default=None,
                    help="T-Rex-format robot YAML (default: deploy/robot/vega/config/default.yaml): camera IPs/ports, "
                         "hand serials, home pose, obstacles (table_height = measured - 0.04 m)")
    ap.add_argument("--output-dir", type=Path, default=REPO / "runs/sharpa_eval")
    a = ap.parse_args()

    import vega_robot

    token = os.environ.get("AGENTICWAM_DEPLOY_TOKEN") or (a.token_file.read_text().strip() if a.token_file.exists() else "")
    if not token:
        raise SystemExit(f"no token: start the server first (it writes {a.token_file}) or set AGENTICWAM_DEPLOY_TOKEN")
    root = a.output_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)

    client = sharpa_client.SharpaPolicyClient(sharpa_client.AgenticWamClientConfig(
        host=a.host, port=a.port, token=token, instruction=a.instruction, image_format=a.image_format))
    health = client.connect()
    robot_cfg = vega_robot.VegaRobotConfig.from_yaml(a.robot_config or vega_robot.DEFAULT_CONFIG)
    robot = vega_robot.VegaSharpaRobot(robot_cfg)
    limits = sharpa_guard.GuardLimits(row0_max=a.row0_max, step_count=a.step_count, step_reject=a.step_reject)

    try:
        import cv2

        obs = robot.get_observation()
        preview = root / "preview_model_view.jpg"
        cv2.imwrite(str(preview), preview_grid(obs["images"]))
        if not preflight(health, obs, preview):
            logger.info("Aborted before any motion.")
            return 1
        # camera image size check, server side: a 320x224 head frame must be refused (the client side is in preflight)
        logger.info("server resolution guard OK: {}", sharpa_client.check_server_resolution_guard(client, obs, a.instruction))
        if a.preview_only:
            logger.info("--preview-only: nothing moved. Preview at {}", preview)
            return 0

        session = {"server": health, "args": {k: str(v) for k, v in vars(a).items()},
                   "robot": vars(robot.cfg), "guard": vars(limits)}
        (root / "session.json").write_text(json.dumps(session, indent=2, default=str))

        for episode in range(a.episodes):
            logger.info("=" * 70)
            if not a.dry_run:
                logger.info("Moving to the start pose (collision-planned)...")
                robot.move_to_start()
            if input(f"\nEpisode {episode + 1}/{a.episodes}: [Enter] start, 'q' quit: ").strip().lower() == "q":
                break
            client.reset(task=a.instruction)
            guard = sharpa_guard.SharpaGuard(limits, joint_limits=robot.joint_limits)
            recorder = EpisodeRecorder(root, episode, {"instruction": a.instruction, "episode": episode,
                                                       "started": dt.datetime.now().isoformat(), **session},
                                       save_obs=not a.no_save_obs)
            result = run_episode(a, robot, client, guard, recorder, a.instruction)
            path = recorder.finish(result)
            g = result["guard"]
            logger.info("Episode saved to {} | {} | {} replans, {} steps | guard rejects {} step hits {} "
                        "limit hits {} | overruns {:.1%}", path, result["stop_reason"], result["replans"],
                        result["steps"], g["rejected"], g["step_hits"], g["limit_hits"],
                        result["control"]["overrun_rate"])
            if result["stop_reason"] in {"aborted", "inference_failed", "command_thread_stopped"}:
                break
    finally:
        robot.close()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
