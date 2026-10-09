#!/usr/bin/env python3
"""Run AGENTICWAM-Cosmos policy evaluation on the YAM station.

    # look at what the model will see -- NOTHING MOVES
    .venv/bin/python scripts/run_agenticwam_eval.py --instruction "put the plastic bottles in the bin" --preview-only

    # run it
    .venv/bin/python scripts/run_agenticwam_eval.py --instruction "put the plastic bottles in the bin"

Control loop is **sequential**: capture -> POST /act (~1.0 s, the arm holds) ->
execute the chunk at 30 Hz -> repeat. Because the arm is stationary while the
model thinks, chunk row 0 genuinely corresponds to the measured state, so no
latency compensation is needed. See docs/REALWORLD.md §4.6.

Everything the model consumes is printed and previewed before the arms are
allowed to move. The two things that silently ruin an eval -- a camera framed
differently than training, and an instruction that does not match the checkpoint
-- are the two things this makes hardest to get wrong.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import datetime as dt
import json
from pathlib import Path
import select
import sys
import time

from loguru import logger
import numpy as np
import tyro

from yam_station.robot import hardware, obs_pipeline, safety, station, agenticwam_client


@dataclass
class EvalConfig:
    instruction: str
    """Language instruction sent to the policy. THE task specification.

    Must describe a task the checkpoint was trained on -- this one is the ABC
    bottles run, so "put the plastic bottles in the bin" is the trained phrasing.
    A novel instruction is an OOD test, not a deployment.
    """

    server_host: str = "127.0.0.1"
    server_port: int = 8712
    token_file: Path = Path("/tmp/agenticwam_token.txt")

    exec_rows: int = 32
    """Rows of each 32-row chunk to execute before replanning.

    32 = maximum throughput. Lower = fresher observations and better closed-loop
    correction, at the cost of one ~1 s planning stall per replan. Worth an A/B.
    """

    control_hz: float = 30.0
    """Rate the chunk rows are played back at. 30 Hz is the training rate."""

    max_replans: int = 20
    """Episode budget. 20 x 32 rows @ 30 Hz ~= 21 s of motion, ~42 s wall clock."""

    episodes: int = 1
    max_joint_delta: float = hardware.MAX_JOINT_DELTA_PER_STEP
    """Per-step joint motion cap, radians. The primary safety parameter."""

    output_dir: Path = Path("~/agenticwam_runs").expanduser()
    session: str = ""
    """Session name. Default: <date>_<port>. One session per server config."""

    preview_only: bool = False
    """Capture, save the model-view preview, print the preflight, and exit.
    Never touches the arms. Run this first, every time."""

    dry_run: bool = False
    """Run the full loop -- capture, infer, safety-filter, record every artifact --
    but never command the arms. Validates the whole path end to end, including
    what the policy wants to do, with zero motion. The safety report from a dry
    run tells you whether the policy's commands are inside the envelope before
    you let them reach hardware."""

    yes: bool = False
    """Skip the confirmation prompt. For unattended runs only."""

    save_obs: bool = True
    """Save the frames actually sent, per replan."""

    start_pose: str = "training_mean"
    """Where each episode begins. `training_mean` = the training set's mean pose
    (recommended); `zeros` = the station's mechanical home, which is far outside
    the training distribution and makes the policy lunge on the first action.

    Better still, once you have one: a pose recorded from an actual episode start.
    """

    home_before: bool = True
    home_after: bool = True


CAMERA_ROLES = ("top", "left", "right")
"""Only the three the policy consumes. `bottom` exists on the station but the
model has no input for it."""

# Mean state over the ABC bottles training set (dataset_stats_deltajoint.json,
# state.default.global_mean). NOT all-zeros -- and that difference matters.
#
# Starting an episode at the station's all-zeros home puts the arm ~1.3 rad away
# from anything in the training data on joints 1 and 2. The policy's first action
# is then a lunge toward a task-typical pose: a dry run from zeros asked for
# 2.2 rad/step (44x the safety limit) and every single step got clamped.
# Starting inside the training distribution is not cosmetic, it is the difference
# between the policy continuing a trajectory and the policy trying to teleport.
TRAINING_MEAN_POSE = {
    "left_joint_pos": [-0.432, 1.295, 1.162, -0.353, -0.065, -0.488],
    "left_gripper_pos": [0.848],
    "right_joint_pos": [0.392, 1.384, 1.274, -0.382, -0.080, 0.289],
    "right_gripper_pos": [0.811],
}


def _poll_stdin() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        return line.strip().lower() if line else None
    return None


class EpisodeRecorder:
    """Writes the per-episode artifact tree described in docs/REALWORLD.md §4.7."""

    def __init__(self, root: Path, episode: int, cfg: EvalConfig, health: dict):
        self.dir = root / f"ep_{episode:03d}"
        (self.dir / "code").mkdir(parents=True, exist_ok=True)
        if cfg.save_obs:
            (self.dir / "obs").mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.started = dt.datetime.now(tz=dt.timezone.utc)
        self._steps = (self.dir / "steps.jsonl").open("w")
        (self.dir / "meta.json").write_text(
            json.dumps(
                {
                    "instruction": cfg.instruction,
                    "episode": episode,
                    "started_utc": self.started.isoformat(),
                    "exec_rows": cfg.exec_rows,
                    "control_hz": cfg.control_hz,
                    "max_joint_delta": cfg.max_joint_delta,
                    "cameras": {r: obs_pipeline.MODEL_W for r in CAMERA_ROLES},
                    "server": health,
                    "camera_rig_note": "record any physical camera change here",
                },
                indent=2,
            )
        )

    def replan(self, index: int, code: str | None, frames: dict, info: dict) -> None:
        if code:
            (self.dir / "code" / f"replan_{index:03d}.txt").write_text(code)
        if self.cfg.save_obs:
            import cv2

            grid = obs_pipeline.preview_grid(frames)
            cv2.imwrite(str(self.dir / "obs" / f"replan_{index:03d}.jpg"), grid)
        (self.dir / "code" / f"replan_{index:03d}.timing.json").write_text(
            json.dumps({k: v for k, v in info.items() if k != "code"}, indent=2)
        )

    def step(self, **row) -> None:
        self._steps.write(json.dumps(row) + "\n")

    def finish(self, result: dict) -> Path:
        self._steps.close()
        (self.dir / "result.json").write_text(json.dumps(result, indent=2))
        return self.dir


def preflight(cfg: EvalConfig, health: dict, frames: dict, preview_path: Path) -> bool:
    """Print exactly what the model will receive, and require a human to agree."""
    bar = "=" * 74
    print(f"\n{bar}\n  AGENTICWAM-COSMOS EVAL PREFLIGHT\n{bar}")
    print(f"  instruction   {cfg.instruction!r}")
    print(f"  server        http://{cfg.server_host}:{cfg.server_port}")
    print(f"                task={health.get('task')}")
    print(
        f"                code_expert={health.get('code_expert')} "
        f"horizon={health.get('action_horizon')} nis={health.get('num_inference_steps')}"
    )
    print("\n  cameras -> what the model sees")
    for role in CAMERA_ROLES:
        frame = frames[agenticwam_client.CAMERA_MAP[role]]
        rep = obs_pipeline.check_aspect(frame.shape[1], frame.shape[0])
        flag = "matches 4:3 training rig" if rep.matches_training else "!! ASPECT MISMATCH !!"
        print(f"    {agenticwam_client.CAMERA_MAP[role]:<13s} <- {role:<6s} {rep.describe()}")
        print(f"    {'':<13s}    {flag}")
    print(f"\n  model input   {obs_pipeline.MODEL_W}x{obs_pipeline.MODEL_H} RGB x 3 views")
    print(f"  PREVIEW       {preview_path}")
    print("                ^ open this and confirm the scene looks like training data")
    print("\n  state         14-D [Ljoint0-5, Lgrip, Rjoint0-5, Rgrip], gripper 0..1 (verified)")
    print("  action        absolute qpos from the server; commanded directly")
    print(
        f"\n  motion        {cfg.exec_rows} rows @ {cfg.control_hz:.0f} Hz "
        f"= {cfg.exec_rows / cfg.control_hz:.2f} s per replan, arm HOLDS ~1.0 s while planning"
    )
    pose = start_pose_target(cfg)
    print(f"  start pose    {cfg.start_pose}  L{[round(float(x),2) for x in pose['left_joint_pos']]}")
    if cfg.start_pose == "zeros":
        print("                !! outside the training distribution -- expect heavy clamping !!")
    print(f"  budget        {cfg.max_replans} replans max")
    print(f"  safety        <= {cfg.max_joint_delta:.3f} rad/step, joint limits clamped, hold on failure")

    warnings = obs_pipeline.validate_cameras(frames)
    if warnings:
        print("\n  WARNINGS")
        for w in warnings:
            print(f"    - {w}")
    print(bar)

    if cfg.yes:
        return True
    return input("  Proceed?  [y/N] ").strip().lower() == "y"


def start_pose_target(cfg: EvalConfig) -> dict:
    """The joint target each episode starts from."""
    if cfg.start_pose == "zeros":
        return {
            "left_joint_pos": np.zeros(6), "left_gripper_pos": np.ones(1),
            "right_joint_pos": np.zeros(6), "right_gripper_pos": np.ones(1),
        }
    return {k: np.asarray(v, dtype=np.float64) for k, v in TRAINING_MEAN_POSE.items()}


def capture_frames(robot: station.YamRobot) -> tuple[dict, dict]:
    """Return (obs, frames-keyed-by-policy-camera-name)."""
    obs = robot.get_observation(with_history=False)
    frames = {agenticwam_client.CAMERA_MAP[r]: obs["images"][r] for r in CAMERA_ROLES}
    return obs, frames


def run_episode(
    cfg: EvalConfig, robot: station.YamRobot, client: agenticwam_client.AgenticWamPolicyClient,
    recorder: EpisodeRecorder, guard: safety.SafetyGuard,
) -> dict:
    client.reset(cfg.instruction)
    if cfg.home_before and not cfg.dry_run:
        robot.move_to(start_pose_target(cfg), duration_s=4.0)
    if cfg.dry_run:
        logger.warning("DRY RUN -- inference and recording only, no arm commands")
    robot.reset_clock()

    logger.info("Running. 's'+Enter = success, 'f' = failure, 'q' = abort.")
    stop_reason, step_index = "max_replans", 0
    start = time.monotonic()

    for replan in range(cfg.max_replans):
        obs, frames = capture_frames(robot)

        # The arm holds its last commanded setpoint through this call -- that is
        # what makes row 0 of the returned chunk correspond to the measured state.
        try:
            chunk, info = client.infer(obs, task=cfg.instruction)
        except agenticwam_client.AgenticWamServerError as exc:
            logger.error("Inference failed: {}", exc)
            stop_reason = "inference_failed"
            break

        recorder.replan(replan, info.get("code"), frames, info)
        logger.info(
            "replan {:2d} | rtt {:5.0f} ms (model {:4.0f}) | rows {}",
            replan, info["round_trip_ms"], info["server_infer_ms"], info["horizon"],
        )

        rows = min(cfg.exec_rows, info["horizon"])
        for row in range(rows):
            state = robot.get_state()
            action = {k: v[row] for k, v in chunk.items()}
            safe_action, report = guard.filter(action, state)
            if not cfg.dry_run:
                robot.command(safe_action)

            recorder.step(
                step=step_index, replan=replan, row=row, t_ns=time.monotonic_ns(),
                state={k: np.asarray(v).tolist() for k, v in state.items() if "joint_pos" in k or "gripper" in k},
                action_raw={k: np.asarray(v).tolist() for k, v in action.items()},
                action_sent={k: np.asarray(v).tolist() for k, v in safe_action.items()},
                clamped=not report.clean,
                round_trip_ms=info["round_trip_ms"] if row == 0 else None,
            )
            step_index += 1
            robot.wait_for_next_step()

        command = _poll_stdin()
        if command in {"s", "f", "q"}:
            stop_reason = {"s": "success", "f": "failure", "q": "aborted"}[command]
            logger.info("Operator: {}", stop_reason)
            break

    duration = time.monotonic() - start
    if cfg.home_after and not cfg.dry_run:
        robot.move_to_home(duration_s=4.0)
    if not cfg.dry_run:
        robot.relax()

    return {
        "stop_reason": stop_reason,
        "verdict": {"success": True, "failure": False}.get(stop_reason),
        "replans": replan + 1,
        "steps": step_index,
        "duration_s": duration,
        "latency": client.stats.summary(),
        "safety": {
            "joint_rate_hits": guard.cumulative.joint_rate_hits,
            "gripper_rate_hits": guard.cumulative.gripper_rate_hits,
            "joint_limit_hits": guard.cumulative.joint_limit_hits,
            "non_finite": guard.cumulative.non_finite,
            "max_joint_delta": guard.cumulative.max_joint_delta,
        },
        "control": {"overrun_rate": robot.overrun_rate},
    }


def main(cfg: EvalConfig) -> None:
    session = cfg.session or f"{dt.date.today().isoformat()}_port{cfg.server_port}"
    root = Path(cfg.output_dir).expanduser() / session
    root.mkdir(parents=True, exist_ok=True)

    client = agenticwam_client.AgenticWamPolicyClient(
        agenticwam_client.AgenticWamClientConfig(
            host=cfg.server_host, port=cfg.server_port,
            token=cfg.token_file.read_text().strip(), instruction=cfg.instruction,
        )
    )
    health = client.connect()

    robot = station.YamRobot(
        station.RobotConfig(camera_roles=CAMERA_ROLES, history_len=1, control_hz=cfg.control_hz)
    )

    try:
        _obs, frames = capture_frames(robot)
        preview = obs_pipeline.save_preview(frames, root / "preview_model_view.jpg")

        if not preflight(cfg, health, frames, preview):
            logger.info("Aborted before any motion.")
            return
        if cfg.preview_only:
            logger.info("--preview-only: nothing moved. Preview at {}", preview)
            return

        (root / "session.json").write_text(
            json.dumps({"session": session, "server": health, "config": {
                "instruction": cfg.instruction, "exec_rows": cfg.exec_rows,
                "control_hz": cfg.control_hz, "max_joint_delta": cfg.max_joint_delta}}, indent=2)
        )

        summary_path = root / "summary.csv"
        new_file = not summary_path.exists()
        with summary_path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["episode", "instruction", "verdict", "stop_reason", "replans",
                                 "steps", "duration_s", "rtt_p50_ms", "clamps"])

            for episode in range(cfg.episodes):
                logger.info("=" * 70)
                logger.info("Episode {}/{}: {!r}", episode + 1, cfg.episodes, cfg.instruction)
                recorder = EpisodeRecorder(root, episode, cfg, health)
                guard = safety.SafetyGuard(safety.SafetyLimits(max_joint_delta=cfg.max_joint_delta))
                result = run_episode(cfg, robot, client, recorder, guard)
                path = recorder.finish(result)

                writer.writerow([
                    episode, cfg.instruction, result["verdict"], result["stop_reason"],
                    result["replans"], result["steps"], round(result["duration_s"], 1),
                    round(result["latency"]["round_trip_ms"]["p50"]),
                    result["safety"]["joint_rate_hits"] + result["safety"]["joint_limit_hits"],
                ])
                fh.flush()
                logger.info("Episode saved to {}", path)
                logger.info(
                    "  {} | {} replans, {} steps, {:.1f} s | clamps j{} g{}",
                    result["stop_reason"], result["replans"], result["steps"],
                    result["duration_s"], result["safety"]["joint_rate_hits"],
                    result["safety"]["gripper_rate_hits"],
                )
                if result["stop_reason"] in {"aborted", "inference_failed"}:
                    break
                if episode + 1 < cfg.episodes and input("\n[Enter] next episode, 'q' to stop: ").strip().lower() == "q":
                    break
    finally:
        robot.close()
        client.close()


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
