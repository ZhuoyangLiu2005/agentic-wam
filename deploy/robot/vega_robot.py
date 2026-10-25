"""vega + Sharpa robot facade for the deploy client -- the counterpart of YAM's `yam_station.robot.station.YamRobot`.

Built from T-Rex's own robot code (vendored in `deploy/robot/vega/`); setup mirrors T-Rex
`hardware_code/eval/eval_trex_async.py:562-708` so the robot is driven exactly as T-Rex drives it:

    YamRobot                     VegaSharpaRobot
    get_observation()  frames    HeadCameraReceiver / WristCameraReceiver (ZMQ, RGB 640x360)
    get_state()                  dexcontrol get_joint_pos_dict + Sharpa get_states().angles -> 58-D
    command(action)              write the row into the action buffer; the 300 Hz full_robot_action_loop
                                 thread sends it through SmoothingAndSafetyManager (0.4 x URDF velocity cap,
                                 URDF limit clip, self + environment collision fallback)
    wait_for_next_step()         loop_rate_limiters.RateLimiter(command_hz)
    move_to(start pose)          move_to_start(): move_robot_to_position_safe (OMPL, 0.3 rad/s)
    relax() / move_to_home()     hold(): clear the buffer, so the arms keep the last target and nothing more
                                 is sent (T-Rex eval behaviour; the operator resets by hand)

State/action layout: [L_arm 7, L_hand 22, R_arm 7, R_hand 22], radians, the dataset's observation.state order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time

from loguru import logger
import numpy as np

STATE_DIM = 58
HAND_JOINT_COUNT = 22
L_ARM, L_HAND, R_ARM, R_HAND = slice(0, 7), slice(7, 29), slice(29, 36), slice(36, 58)

# contract camera key -> (receiver, stream name)
CAMERA_STREAMS = {"head_camera": ("head", "HEAD_CAM"), "left_camera": ("wrist", "LEFT_WRIST"),
                  "right_camera": ("wrist", "RIGHT_WRIST")}


DEFAULT_CONFIG = Path(__file__).resolve().parent / "vega" / "config" / "default.yaml"


@dataclass
class VegaRobotConfig:
    """Robot-station settings. Built from a T-Rex-format YAML (`from_yaml`); the default file is
    deploy/robot/vega/config/default.yaml (copied from T-Rex hardware_code/config/default.yaml)."""

    left_arm_home: list[float]
    right_arm_home: list[float]
    torso: list[float]
    head: list[float]
    left_hand_serial: str
    right_hand_serial: str
    hand_interpolate: bool
    image_width: int
    image_height: int
    head_camera_ip: str
    head_camera_ports: dict[str, int]       # {"HEAD_CAM": port}
    wrist_camera_ip: str
    wrist_camera_ports: dict[str, int]      # {"LEFT_WRIST": port, "RIGHT_WRIST": port}
    command_hz: float
    arm_action_hz: float
    reset_dof_err_tol: float
    # Collision obstacles (m; None disables one). Measure at your site: table = measured height - 0.04.
    table_height: float | None
    back_wall_distance: float | None
    left_wall_distance: float | None
    right_wall_distance: float | None
    source: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> "VegaRobotConfig":
        import yaml

        y = yaml.safe_load(Path(path).read_text())
        r, h, c, ctl, env = y["robot"], y["hands"], y["cameras"], y["control"], y["environment"]
        for name, ports in (("head", c["head"]["ports"]), ("wrist", c["wrist"]["ports"])):
            want = {"head": {"HEAD_CAM"}, "wrist": {"LEFT_WRIST", "RIGHT_WRIST"}}[name]
            if set(ports) != want:
                raise ValueError(f"{path}: cameras.{name}.ports must have exactly {sorted(want)}, got {sorted(ports)}")
        return cls(
            left_arm_home=list(r["left_arm_default_joint_pos"]), right_arm_home=list(r["right_arm_default_joint_pos"]),
            torso=list(r["torso_default_joint_pos"]), head=list(r["head_default_joint_pos"]),
            left_hand_serial=str(h["left_serial"]), right_hand_serial=str(h["right_serial"]),
            hand_interpolate=bool(h["interpolate"]),
            image_width=int(c["image_width"]), image_height=int(c["image_height"]),
            head_camera_ip=str(c["head"]["sender_ip"]), head_camera_ports=dict(c["head"]["ports"]),
            wrist_camera_ip=str(c["wrist"]["sender_ip"]), wrist_camera_ports=dict(c["wrist"]["ports"]),
            command_hz=float(ctl["command_hz"]), arm_action_hz=float(ctl["arm_action_hz"]),
            reset_dof_err_tol=float(ctl["reset_dof_err_tol"]),
            table_height=env["table_height"], back_wall_distance=env["back_wall_distance"],
            left_wall_distance=env["left_wall_distance"], right_wall_distance=env["right_wall_distance"],
            source=str(Path(path).resolve()),
        )

    @property
    def home(self) -> dict[str, np.ndarray]:
        return {"left_arm": np.array(self.left_arm_home), "right_arm": np.array(self.right_arm_home),
                "torso": np.array(self.torso), "head": np.array(self.head)}


def connect_hands(left_serial: str, right_serial: str):
    """T-Rex eval_trex_async.connect_hands."""
    from sharpa import SharpaWaveManager

    manager = SharpaWaveManager.get_instance()
    time.sleep(1)  # device discovery
    devices = manager.get_all_device_sn()
    logger.info("Available hand devices: {}", devices)
    for side, serial in (("left", left_serial), ("right", right_serial)):
        if serial not in devices:
            raise RuntimeError(f"{side} hand serial {serial} not found in devices: {devices}")
    return manager.connect(left_serial), manager.connect(right_serial)


def initialize_hand(hand) -> None:
    """T-Rex eval_trex_async.initialize_hand: position mode, speed 0.3, current 0.6, SDK control."""
    from sharpa import ControlMode, ControlSource

    for name, call in (("control mode", lambda: hand.set_control_mode(ControlMode.POSITION)),
                       ("speed coeff", lambda: hand.set_speed_coeff(0.3)),
                       ("current coeff", lambda: hand.set_current_coeff(0.6)),
                       ("control source", lambda: hand.set_control_source(ControlSource.SDK))):
        error = call()
        if error.code != 0:
            raise RuntimeError(f"Failed to set {name}: {error.message}")


class VegaSharpaRobot:
    """Arms (dexcontrol), hands (Sharpa Wave), cameras (ZMQ) and the 300 Hz command thread."""

    def __init__(self, cfg: VegaRobotConfig | None = None):
        from dexcontrol.robot import Robot
        from loop_rate_limiters import RateLimiter

        from vega.arm_hand_control import (InitializationCollisionPlanner, SmoothingAndSafetyManager,
                                           full_robot_action_loop)
        from vega.head_camera_receiver import HeadCameraReceiver
        from vega.robot_descriptions import (DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES, add_env_obstacles,
                                             build_full_robot)
        from vega.wrist_camera_receiver import WristCameraReceiver

        self.cfg = cfg = cfg or VegaRobotConfig.from_yaml()
        logger.info("Robot config: {}", cfg.source or "(in code)")
        self._joint_names = DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES
        self._RateLimiter = RateLimiter
        self._closed = False

        logger.info("Initializing robot...")
        self.robot = Robot()
        logger.info("Robot '{}' initialized", self.robot.robot_model)

        self.head_cam = HeadCameraReceiver(im_h=cfg.image_height, im_w=cfg.image_width,
                                           sender_ip=cfg.head_camera_ip, ports=cfg.head_camera_ports)
        self.head_cam.start_receiving(timeout=10.0)
        self.wrist_cam = WristCameraReceiver(im_h=cfg.image_height, im_w=cfg.image_width,
                                             sender_ip=cfg.wrist_camera_ip,
                                             ports=cfg.wrist_camera_ports)
        self.wrist_cam.start_receiving(timeout=10.0)
        self._receivers = {"head": self.head_cam, "wrist": self.wrist_cam}

        home = cfg.home
        wrapper, assemble_qpos, disassemble_qpos = build_full_robot(default_joint_by_component=home)
        wrapper = add_env_obstacles(
            robot=wrapper,
            default_joint_by_component={"left_arm": home["left_arm"], "right_arm": home["right_arm"],
                                        "left_hand": np.zeros(HAND_JOINT_COUNT),
                                        "right_hand": np.zeros(HAND_JOINT_COUNT)},
            assemble_qpos=assemble_qpos,
            back_wall_distance=cfg.back_wall_distance, left_wall_distance=cfg.left_wall_distance,
            right_wall_distance=cfg.right_wall_distance, table_height=cfg.table_height,
        )
        safety = SmoothingAndSafetyManager(pin_full_robot_wrapper=wrapper, assemble_qpos=assemble_qpos,
                                           disassemble_qpos=disassemble_qpos, ruckig_smoothing=False,
                                           action_hz=cfg.arm_action_hz)

        logger.info("Connecting to hands...")
        self.left_hand, self.right_hand = connect_hands(cfg.left_hand_serial, cfg.right_hand_serial)
        for hand in (self.left_hand, self.right_hand):
            initialize_hand(hand)
            hand.start()
        self.left_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, cfg.hand_interpolate)
        self.right_hand.set_joint_position([0.0] * HAND_JOINT_COUNT, cfg.hand_interpolate)
        time.sleep(1)

        self.action_buf_lock = threading.Lock()
        self.action_buffer = {"left_arm": None, "right_arm": None, "left_hand": None, "right_hand": None}
        self.hardware_lock = threading.Lock()
        self._terminate = threading.Event()
        self._thread = threading.Thread(
            target=full_robot_action_loop,
            kwargs={"terminate_event": self._terminate, "action_buf_lock": self.action_buf_lock,
                    "action_buffer": self.action_buffer, "hardware_lock": self.hardware_lock,
                    "dexmate_bimanual_robot": self.robot, "sharpa_left_hand": self.left_hand,
                    "sharpa_right_hand": self.right_hand, "smoothing_and_safety_manager": safety,
                    "action_hz": cfg.arm_action_hz, "hand_interpolate": cfg.hand_interpolate},
            daemon=True,
        )
        self._thread.start()
        logger.info("Full robot action thread started at {} Hz", cfg.arm_action_hz)

        lo = disassemble_qpos(np.asarray(wrapper.model.lowerPositionLimit, dtype=np.float64))
        hi = disassemble_qpos(np.asarray(wrapper.model.upperPositionLimit, dtype=np.float64))
        pack = lambda d: np.concatenate([d["left_arm"], d["left_hand"], d["right_arm"], d["right_hand"]])
        self.joint_limits = (pack(lo), pack(hi))  # 58-D, the same limits the 300 Hz thread clips to

        assert 0.3 / cfg.command_hz < 0.05  # T-Rex's own sanity check on the reset speed
        self._planner = InitializationCollisionPlanner(
            pin_full_robot_wrapper=wrapper, assemble_qpos=assemble_qpos, disassemble_qpos=disassemble_qpos,
            max_edge_joint_step=0.3 / cfg.command_hz, plan_timeout_s=10.0, solve_step_s=0.1)
        self.reset_clock()

    # --- observation -----------------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """58-D [L_arm 7, L_hand 22, R_arm 7, R_hand 22] (rad)."""
        with self.hardware_lock:
            arm = self.robot.get_joint_pos_dict(component=["left_arm", "right_arm"])
            left_hand = np.array(self.left_hand.get_states().angles, dtype=np.float64)
            right_hand = np.array(self.right_hand.get_states().angles, dtype=np.float64)
        left_arm = np.array([arm[n] for n in self._joint_names["left_arm"]], dtype=np.float64)
        right_arm = np.array([arm[n] for n in self._joint_names["right_arm"]], dtype=np.float64)
        return np.concatenate([left_arm, left_hand, right_arm, right_hand])

    def get_observation(self) -> dict:
        """{"images": {contract camera key: RGB 360x640x3}, "state": (58,), "capture_ns", "frame_age_s"}."""
        state = self.get_state()  # state first, then frames (T-Rex eval order)
        now = time.time()
        images, ages = {}, {}
        for key, (receiver, stream) in CAMERA_STREAMS.items():
            frame = self._receivers[receiver].get_data(stream)
            images[key] = frame.image_rgb
            ages[key] = now - frame.ts_sec
        return {"images": images, "state": state, "capture_ns": time.monotonic_ns(), "frame_age_s": ages}

    # --- commanding ------------------------------------------------------------------------------

    def command(self, row: np.ndarray) -> None:
        """Hand one absolute 58-D target to the 300 Hz thread (which clips, rate-limits, collision-checks)."""
        row = np.asarray(row, dtype=np.float64).reshape(STATE_DIM)
        with self.action_buf_lock:
            self.action_buffer["left_arm"] = row[L_ARM].copy()
            self.action_buffer["left_hand"] = row[L_HAND].copy()
            self.action_buffer["right_arm"] = row[R_ARM].copy()
            self.action_buffer["right_hand"] = row[R_HAND].copy()

    def hold(self) -> None:
        """Stop sending: the arms keep the last target they were given."""
        with self.action_buf_lock:
            for k in self.action_buffer:
                self.action_buffer[k] = None

    @property
    def command_thread_alive(self) -> bool:
        """False once the 300 Hz thread has stopped itself (tracking-error safety stop)."""
        return self._thread.is_alive() and not self._terminate.is_set()

    def move_to_start(self) -> None:
        """Collision-planned move to the home pose (arms), hands open, torso/head at the dataset values."""
        from vega.arm_hand_control import move_robot_to_position_safe

        self.hold()
        home = self.cfg.home
        move_robot_to_position_safe(
            dexmate_bimanual_robot=self.robot, sharpa_left_hand=self.left_hand, sharpa_right_hand=self.right_hand,
            initialization_planner=self._planner, action_buffer=self.action_buffer,
            action_buf_lock=self.action_buf_lock,
            target_joint_pos={"left_arm": home["left_arm"], "right_arm": home["right_arm"],
                              "left_hand": np.zeros(HAND_JOINT_COUNT), "right_hand": np.zeros(HAND_JOINT_COUNT),
                              "head": home["head"], "torso": home["torso"]},
            hardware_lock=self.hardware_lock, command_hz=self.cfg.command_hz,
            dof_error_tolerance=self.cfg.reset_dof_err_tol, hold_time_s=0.5,
        )
        time.sleep(1)

    # --- timing ----------------------------------------------------------------------------------

    def reset_clock(self) -> None:
        self._limiter = self._RateLimiter(frequency=self.cfg.command_hz, name="sharpa_eval_loop", warn=False)
        self._steps = self._overruns = 0
        self._deadline = None

    def wait_for_next_step(self) -> None:
        # Own deadline bookkeeping: RateLimiter.remaining is a property in some loop-rate-limiters
        # versions and a method in others.
        self._steps += 1
        if self._deadline is not None and time.perf_counter() > self._deadline:
            self._overruns += 1
        self._limiter.sleep()
        self._deadline = time.perf_counter() + 1.0 / self.cfg.command_hz

    @property
    def overrun_rate(self) -> float:
        return self._overruns / max(1, self._steps)

    # --- shutdown --------------------------------------------------------------------------------

    def close(self) -> None:
        """T-Rex eval shutdown order (plus the head receiver, which T-Rex leaves running)."""
        if self._closed:
            return
        self._closed = True
        from sharpa import SharpaWaveManager

        self.hold()
        self._terminate.set()
        self._thread.join(timeout=2.0)
        self.wrist_cam.stop()
        self.head_cam.stop()
        self.left_hand.stop()
        self.right_hand.stop()
        SharpaWaveManager.get_instance().disconnect_all()
        self.robot.shutdown()
        logger.info("Robot shutdown complete")
