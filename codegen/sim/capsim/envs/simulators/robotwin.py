"""Low-level :class:`BaseEnv` wrapper around a RoboTwin ``Base_Task``: reset, observations, success and episode recording.

Observations are per-camera dicts (``obs[cam]["images"]["rgb"|"depth"]``, ``["intrinsics"]``, ``["pose_mat"]``)
keyed by RoboTwin camera name; manipulation goes through the skill APIs, not :meth:`RoboTwinEnv.step`.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any

import numpy as np

from capsim.envs.base import BaseEnv
from capsim.integrations.robotwin import load_robotwin_task
from capsim.profiles import DEFAULT_EMBODIMENT, load_profile

# (save_dir, ep_num) already finalized by this process: reset, atexit and duplicate instances may each try.
_FINALIZED_EPISODES: set[tuple[str, int]] = set()

class RoboTwinEnv(BaseEnv):
    """Franka-style cap-x wrapper for a single RoboTwin dual-arm task."""

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_clean",
        embodiment: str = DEFAULT_EMBODIMENT,
        privileged: bool = True,
        max_steps: int = 1000,
        seed: int | None = None,
        enable_render: bool = False,
        viser_debug: bool = False,
        camera_depth: bool = True,
        planner: str | None = None,
        save_episode_data: bool = False,
        save_dir: str | None = None,
        save_freq: int | None = None,
        record_instructions: str | bool = "static",
        n_instructions: int = 100,
    ) -> None:
        super().__init__()
        self.profile = load_profile(embodiment)
        self._primary_camera = str(self.profile.cameras.primary)
        if save_freq is None:
            save_freq = int(self.profile.recording.save_freq)
        # Instruction placeholders of recorded episodes: "static" evaluates the task's info literal after the program
        # ran (no extra rollout); "expert" runs RoboTwin's expert first (tasks with several arm variables); "off".
        mode = {True: "static", False: "off"}.get(record_instructions, str(record_instructions))
        if mode not in ("static", "expert", "off"):
            raise ValueError(f"record_instructions must be static | expert | off, got {record_instructions!r}")
        self.record_instructions = mode if save_episode_data else "off"
        self.n_instructions = int(n_instructions)
        self._episode_language: dict[str, Any] | None = None
        self.privileged = privileged
        self.task_name = task_name
        self.task_config = task_config
        self.embodiment = embodiment
        self.max_steps = max_steps
        self.seed = seed
        self.enable_render = enable_render
        self.viser_debug = viser_debug  # unused; kept for interface parity
        self._camera_depth = camera_depth

        # RoboTwin's native per-step saver: per-frame obs pkls merged into an episode hdf5.
        self.save_episode_data = save_episode_data
        self._save_dir_abs: str | None = None
        extra_args = None
        if save_episode_data:
            if not save_dir:
                raise ValueError("save_episode_data requires save_dir")
            self._save_dir_abs = os.path.abspath(save_dir)
            os.makedirs(self._save_dir_abs, exist_ok=True)
            # Per-frame pkls fail on NFS in long rollouts: .cache goes to host-local scratch, so one host per collection.
            self._setup_cache_scratch(self._save_dir_abs)
            extra_args = {
                "save_data": True,
                "save_path": self._save_dir_abs,
                "save_freq": save_freq,
            }

        self.handle = load_robotwin_task(
            task_name=task_name,
            task_config=task_config,
            embodiment=embodiment,
            camera_depth=camera_depth,
            render_freq=10 if enable_render else 0,
            planner=planner,
            extra_args=extra_args,
        )

        self._sim_step_count = 0
        self._reset_seed = 0
        self._current_obs: dict[str, Any] | None = None

        # Code-line alignment records of the current episode, appended by the API's motion executor.
        self.code_alignment_buffer: list[dict[str, Any]] = []

        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []

        if save_episode_data:
            # The last episode has no following reset() to finalize it.
            atexit.register(self._finalize_episode)

        self.reset(seed=seed)

    @staticmethod
    def _setup_cache_scratch(save_dir_abs: str) -> None:
        """Symlink {save_dir}/.cache to host-local scratch; idempotent under concurrent workers on one host."""
        import getpass
        import tempfile

        cache_link = os.path.join(save_dir_abs, ".cache")
        scratch_root = os.environ.get("CAPX_ROLLOUT_CACHE")
        if not scratch_root:
            local = f"/data/local/{getpass.getuser()}"
            base = local if os.path.isdir("/data/local") else tempfile.gettempdir()
            scratch_root = os.path.join(base, "_rollout_cache")
        # Keyed by the full save_dir: basenames ("clean"/"randomized") repeat across collections on one host.
        scratch = os.path.join(
            scratch_root, save_dir_abs.rstrip("/").lstrip("/").replace("/", "__")
        )
        os.makedirs(scratch, exist_ok=True)
        if os.path.exists(cache_link) and not os.path.islink(cache_link):
            return
        try:
            if os.path.islink(cache_link):
                if os.path.realpath(cache_link) == os.path.realpath(scratch):
                    return  # already linked correctly
                os.unlink(cache_link)
            os.symlink(scratch, cache_link)
        except FileExistsError:
            pass  # another worker created it first

    # ----------------------- Core BaseEnv interface -----------------------

    @property
    def task(self):
        """The underlying RoboTwin ``Base_Task`` instance."""
        return self.handle.task

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from envs.utils.create_actor import UnStableError  # type: ignore[import-not-found]

        # Finalize the previous episode while its end state still exists (success is judged on it).
        self._finalize_episode()

        base_seed = 0 if seed is None else int(seed)
        if self.save_episode_data:
            # Stale pkls from a killed run would be spliced into this episode's hdf5.
            import shutil

            shutil.rmtree(
                os.path.join(self._save_dir_abs, ".cache", f"episode{base_seed}"),
                ignore_errors=True,
            )
        last_err: Exception | None = None
        for attempt in range(20):
            cur_seed = base_seed + attempt
            try:
                # Unstable scenes retry seed+1, but the episode is keyed by the requested seed (one episode per seed).
                self.handle.setup(
                    seed=cur_seed,
                    ep_num=base_seed if self.save_episode_data else 0,
                )
                self._reset_seed = base_seed
                self._scene_seed = cur_seed  # actual (possibly shifted) scene seed
                if cur_seed != base_seed and self.save_episode_data:
                    print(
                        f"[robotwin.reset] seed {base_seed} unstable; "
                        f"recording nearest stable scene {cur_seed} under ep {base_seed}"
                    )
                break
            except UnStableError as exc:  # unstable init; try the next seed
                last_err = exc
                self.handle.close()
                continue
        else:
            raise RuntimeError(
                f"RoboTwin task '{self.task_name}' could not produce a stable scene "
                f"near seed {base_seed}"
            ) from last_err

        self._episode_language = None
        if self.record_instructions == "expert":
            self._episode_language = self._harvest_instructions(cur_seed, base_seed)

        self._sim_step_count = 0
        step_lim = getattr(self.task, "step_lim", None)
        if step_lim:
            self.max_steps = int(step_lim)

        self._install_frame_hook()

        obs = self.get_observation()
        if self._record_frames:
            self.record_video_frame()
        info = {"task_prompt": self.handle.instruction, "seed": self._reset_seed}
        return obs, info

    def _harvest_instructions(self, scene_seed: int, ep_num: int) -> dict[str, Any]:
        """Run the expert for the instruction placeholders, then rebuild the scene; failure keeps the template."""
        import shutil

        lang: dict[str, Any] = {"placeholders": {}, "instructions": {"seen": [], "unseen": []}, "expert_success": None}
        try:
            # Non-recording scene for the expert; the recording setup's cache is discarded below.
            self.handle.close()
            self.handle.setup(seed=scene_seed, ep_num=0, save_data=False)
            ph = self.handle.expert_placeholders()
            lang["placeholders"] = ph
            try:
                lang["expert_success"] = bool(self.task.check_success())
            except Exception:
                pass
            if ph:
                lang["instructions"] = self.handle.episode_descriptions(ph, self.n_instructions)
        except Exception as exc:  # noqa: BLE001
            print(f"[robotwin.reset] instruction harvest failed for seed {scene_seed}: {exc!r} — keeping the template")
        finally:
            self.handle.close()
            shutil.rmtree(os.path.join(self._save_dir_abs, ".cache", f"episode{ep_num}"), ignore_errors=True)
            self.handle.setup(seed=scene_seed, ep_num=ep_num)
        return lang

    def _static_instructions(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Fill placeholders from the task's info literal on the finished episode, plus RoboTwin paraphrases."""
        from capsim.integrations.robotwin import placeholders as P

        lang: dict[str, Any] = {"placeholders": {}, "instructions": {"seen": [], "unseen": []}, "source": "static"}
        res = P.evaluate(self.task, P.program_hands(records))
        lang.update({"arm_vars": res["arm_vars"], "needs_expert": res["needs_expert"], "error": res["error"]})
        if res["placeholders"]:
            lang["placeholders"] = res["placeholders"]
            try:
                lang["instructions"] = self.handle.episode_descriptions(res["placeholders"], self.n_instructions)
            except Exception as exc:  # noqa: BLE001
                lang["error"] = f"description generation failed: {exc!r}"
        elif res["needs_expert"]:
            print(f"[robotwin] {self.task_name}: info uses arm variables {res['arm_vars']} -> use record_instructions=expert")
        return lang

    def _finalize_episode(self) -> None:
        """Merge the episode's pkls into hdf5 + video and write its alignment file and manifest line (if recorded)."""
        if not self.save_episode_data:
            return
        num_frames = int(getattr(self.task, "FRAME_IDX", 0) or 0)
        records = self.code_alignment_buffer
        self.code_alignment_buffer = []
        if num_frames <= 0:
            return
        key = (self._save_dir_abs, int(getattr(self.task, "ep_num", self._reset_seed)))
        if key in _FINALIZED_EPISODES:
            return
        _FINALIZED_EPISODES.add(key)
        from capsim.integrations.robotwin.code_alignment import flush_episode_alignment

        try:
            success = bool(self.task.check_success())
        except Exception:
            success = False
        if self.record_instructions == "static":
            self._episode_language = self._static_instructions(records)
        if getattr(self, "episode_nouns", None):
            self._episode_language = dict(self._episode_language or {})
            self._episode_language["materialized"] = self.episode_nouns
        ep_num = int(getattr(self.task, "ep_num", self._reset_seed))
        self.task.merge_pkl_to_hdf5_video()
        self.task.remove_data_cache()
        self.task.FRAME_IDX = 0  # guard against double-finalize (reset + atexit)
        # Program text: the per-episode materialized template if any, else CAPX_ALIGN_CODE_FILE.
        code_text = getattr(self, "episode_code", None)
        code_file = os.environ.get("CAPX_ALIGN_CODE_FILE")
        if code_text is None and code_file and os.path.exists(code_file):
            with open(code_file, encoding="utf-8") as f:
                code_text = f.read()
        flush_episode_alignment(
            self._save_dir_abs,
            ep_num,
            seed=self._reset_seed,
            success=success,
            instruction=getattr(self.handle, "instruction", None),
            num_frames=num_frames,
            code=code_text,
            records=records,
            language=self._episode_language,
        )
        with open(os.path.join(self._save_dir_abs, "manifest.jsonl"), "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "seed": self._reset_seed,
                        "ep_num": ep_num,
                        "scene_seed": getattr(self, "_scene_seed", self._reset_seed),
                        "success": success,
                        "num_frames": num_frames,
                        "task_config": self.task_config,
                        "task_name": self.task_name,
                    }
                )
                + "\n"
            )

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level fallback step; control normally flows through the skill API."""
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = bool(self.task_completed())
        truncated = self._sim_step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def compute_reward(self) -> float:
        try:
            return 1.0 if self.task_completed() else 0.0
        except Exception:
            return 0.0

    def task_completed(self) -> bool:
        try:
            return bool(self.task.check_success())
        except Exception as exc:  # noqa: BLE001
            # Warn once per episode so an environment defect is not mistaken for a program failure.
            key = (self._reset_seed, type(exc).__name__, str(exc))
            if getattr(self, "_check_success_error", None) != key:
                self._check_success_error = key
                print(
                    f"[robotwin] {self.task_name}: check_success() raised "
                    f"{type(exc).__name__}: {exc} -> reward 0 (environment defect, not the program)",
                    file=sys.stderr,
                    flush=True,
                )
            return False

    # ----------------------- Observation conversion -----------------------

    def get_observation(self) -> dict[str, Any]:
        """Return RoboTwin obs in cap-x's per-camera dict format."""
        raw = self.task.get_obs()
        self._current_obs = raw
        cam_obs = raw.get("observation", {})

        obs: dict[str, Any] = {}
        for cam_name, cam_data in cam_obs.items():
            entry: dict[str, Any] = {"images": {}}
            if "rgb" in cam_data:
                entry["images"]["rgb"] = np.asarray(cam_data["rgb"], dtype=np.uint8)
            if "depth" in cam_data:
                # Profile depth unit (mm) -> (H, W, 1) metres.
                depth_m = np.asarray(cam_data["depth"], dtype=np.float32) / self.profile.depth_divisor
                entry["images"]["depth"] = depth_m[..., None]
            if "intrinsic_cv" in cam_data:
                entry["intrinsics"] = np.asarray(cam_data["intrinsic_cv"], dtype=np.float64)
            if "extrinsic_cv" in cam_data:
                # extrinsic_cv is world->cam (CV); cap-x pose_mat is cam->world.
                extr = np.asarray(cam_data["extrinsic_cv"], dtype=np.float64)
                ext44 = np.eye(4)
                ext44[: extr.shape[0], : extr.shape[1]] = extr
                entry["pose_mat"] = np.linalg.inv(ext44)
            if "cam2world_gl" in cam_data:
                entry["cam2world_gl"] = np.asarray(cam_data["cam2world_gl"], dtype=np.float64)
            obs[cam_name] = entry

        try:
            left_js = self.task.robot.get_left_arm_jointState()
            right_js = self.task.robot.get_right_arm_jointState()
            obs["robot_joint_pos"] = np.array(left_js + right_js, dtype=np.float64)
            obs["left_arm_joint_pos"] = np.array(left_js, dtype=np.float64)
            obs["right_arm_joint_pos"] = np.array(right_js, dtype=np.float64)
            obs["left_endpose"] = np.array(self.task.get_arm_pose("left"), dtype=np.float64)
            obs["right_endpose"] = np.array(self.task.get_arm_pose("right"), dtype=np.float64)
        except Exception:
            pass

        return obs

    def get_current_time_s(self) -> float:
        return float(self._sim_step_count)

    # ------------------------- Video capture -------------------------

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True, **_: Any) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self.record_video_frame()

    def record_video_frame(self) -> None:
        """Capture one head-camera RGB frame on demand (e.g. the initial frame)."""
        if not self._record_frames:
            return
        try:
            self.task._update_render()
            self.task.cameras.update_picture()
            rgb = self.task.cameras.get_rgb()
            frame = rgb[self._primary_camera]["rgb"]
            self._frame_buffer.append(np.asarray(frame, dtype=np.uint8).copy())
        except Exception:
            pass

    def _install_frame_hook(self) -> None:
        """Wrap the task's ``_take_picture`` (called every ``save_freq`` substeps) to also buffer a video frame."""
        env = self
        # With recording on, the native method is the per-step pkl saver and must still run.
        native_take_picture = type(self.task)._take_picture

        def _capture_frame(*_args, **_kwargs):
            if env._record_frames:
                try:
                    env.task._update_render()
                    env.task.cameras.update_picture()
                    rgb = env.task.cameras.get_rgb()[env._primary_camera]["rgb"]
                    env._frame_buffer.append(np.asarray(rgb, dtype=np.uint8).copy())
                except Exception:
                    pass
            if env.save_episode_data:
                native_take_picture(env.task)

        self.task._take_picture = _capture_frame

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [f.copy() for f in self._frame_buffer[start:end]]

    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        self.task._update_render()
        self.task.cameras.update_picture()
        return np.asarray(self.task.cameras.get_rgb()[self._primary_camera]["rgb"], dtype=np.uint8)

    def close(self) -> None:
        self.handle.close()


class RoboTwinTask(RoboTwinEnv):
    """RoboTwin task selected by name from a YAML config::

        env:
          _target_: capsim.envs.simulators.robotwin.RoboTwinTask
          task_name: place_object_basket
          task_config: demo_clean
          embodiment: aloha-agilex
    """

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_clean",
        embodiment: str = DEFAULT_EMBODIMENT,
        privileged: bool = True,
        max_steps: int = 1000,
        seed: int | None = None,
        enable_render: bool = False,
        viser_debug: bool = False,
        planner: str | None = None,
        save_episode_data: bool = False,
        save_dir: str | None = None,
        save_freq: int | None = None,
        record_instructions: bool = True,
        n_instructions: int = 100,
    ) -> None:
        super().__init__(
            task_name=task_name,
            task_config=task_config,
            embodiment=embodiment,
            privileged=privileged,
            max_steps=max_steps,
            seed=seed,
            enable_render=enable_render,
            viser_debug=viser_debug,
            planner=planner,
            save_episode_data=save_episode_data,
            save_dir=save_dir,
            save_freq=save_freq,
            record_instructions=record_instructions,
            n_instructions=n_instructions,
        )


__all__ = ["RoboTwinEnv", "RoboTwinTask"]
