"""Mocked skill / robot API + a safe sandbox to parse and run generated policy code.

The generated policy code calls these primitives. The mock implementations are
pure no-ops returning lightweight handles so code is *executable* for syntax /
control-flow validation without any real robot I/O. `run_policy_code_safely`
validates with `ast.parse` and executes in a restricted namespace with no
builtins — it must never run arbitrary unsafe code.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class _Pose:
    name: str = "pose"
    pregrasp: "Any" = None
    grasp: "Any" = None

    def __post_init__(self):
        if self.pregrasp is None:
            self.pregrasp = _Pose(name=self.name + ".pregrasp", pregrasp=self, grasp=self)
        if self.grasp is None:
            self.grasp = _Pose(name=self.name + ".grasp", pregrasp=self, grasp=self)


@dataclass
class _Obj:
    name: str
    attached: bool = False


@dataclass
class MockObs:
    """Stand-in observation; carries nothing meaningful for the mock."""
    frame_id: int = 0
    data: Dict[str, Any] = field(default_factory=dict)


class _Gripper:
    def __init__(self):
        self.closed = False


class _Arm:
    """One robot arm (left/right). Has its own gripper and motion primitives."""

    def __init__(self, name: str, log: list[str]):
        self.name = name
        self.gripper = _Gripper()
        self._log = log

    def move_to(self, pose):
        self._log.append(f"{self.name}.move_to({getattr(pose, 'name', pose)})")

    def open_gripper(self):
        self.gripper.closed = False
        self._log.append(f"{self.name}.open_gripper")

    def close_gripper(self):
        self.gripper.closed = True
        self._log.append(f"{self.name}.close_gripper")

    def delta_pose(self, dx, dy, dz, droll, dpitch, dyaw):
        self._log.append(f"{self.name}.delta_pose({dx},{dy},{dz},{droll},{dpitch},{dyaw})")

    def execute_trajectory(self, traj):
        self._log.append(f"{self.name}.execute_trajectory")


class MockRobot:
    """Dual-arm mock. `robot.left_arm` / `robot.right_arm` are full arms; the bare
    `robot.move_to(...)` / `robot.gripper` shims delegate to the right arm so older
    single-arm policy code still runs."""

    def __init__(self):
        self.log: list[str] = []
        self.left_arm = _Arm("left_arm", self.log)
        self.right_arm = _Arm("right_arm", self.log)

    # single-arm back-compat shims -> right arm
    @property
    def gripper(self):
        return self.right_arm.gripper

    def move_to(self, pose):
        self.right_arm.move_to(pose)

    def open_gripper(self):
        self.right_arm.open_gripper()

    def close_gripper(self):
        self.right_arm.close_gripper()

    def delta_pose(self, dx, dy, dz, droll, dpitch, dyaw):
        self.right_arm.delta_pose(dx, dy, dz, droll, dpitch, dyaw)

    def execute_trajectory(self, traj):
        self.right_arm.execute_trajectory(traj)


class MockSkills:
    def __init__(self):
        self._objs: Dict[str, _Obj] = {}

    def detect(self, name, obs):
        self._objs.setdefault(name, _Obj(name=name))
        return self._objs[name]

    def track(self, name, obs):
        return self.detect(name, obs)

    def sample_grasp(self, obj):
        return _Pose(name=f"grasp({getattr(obj, 'name', obj)})")

    def sample_place_pose(self, obj, target):
        return _Pose(name=f"place({getattr(obj, 'name', obj)}->{getattr(target, 'name', target)})")

    def place_inside(self, obj, container):
        return _Pose(name=f"place_inside({getattr(obj, 'name', obj)})")

    def is_attached(self, obj, gripper):
        return bool(getattr(obj, "attached", False))

    def check_attached(self, obj, gripper):
        # mock: object becomes attached once the gripper has closed
        if isinstance(obj, _Obj) and getattr(gripper, "closed", False):
            obj.attached = True
        return bool(getattr(obj, "attached", False))

    def verify_inside(self, obj, container):
        return True

    def recover_grasp(self, obj):
        return False

    def visual_diff(self, obs_before, obs_after):
        return {"changed": True}


# Names the sandbox is allowed to reference.
_ALLOWED_NAMES = {"obs", "state", "robot", "skills", "policy_chunk", "policy_step"}


def _validate_ast(code_str: str) -> ast.AST:
    """Parse and reject obviously unsafe constructs (imports, dunder access)."""
    tree = ast.parse(code_str)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("policy code may not import modules")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("policy code may not access dunder attributes")
        if isinstance(node, ast.Name) and node.id in ("eval", "exec", "compile", "open", "__import__"):
            raise ValueError(f"policy code may not reference {node.id}")
    return tree


def policy_code_parses(code_str: str) -> bool:
    """True if the code is valid Python and passes the safety AST checks."""
    try:
        _validate_ast(code_str)
        return True
    except Exception:
        return False


def run_policy_code_safely(code_str: str, obs=None, state=None, robot=None, skills=None) -> Dict[str, Any]:
    """ast.parse + safety-validate, then exec in a restricted namespace and call
    policy_chunk/policy_step against the mock API. Returns {ok, result, error, robot_log}.
    Never executes with real builtins or I/O."""
    out: Dict[str, Any] = {"ok": False, "result": None, "error": None, "robot_log": []}
    try:
        _validate_ast(code_str)
    except Exception as exc:
        out["error"] = f"parse/validate: {exc}"
        return out

    robot = robot if robot is not None else MockRobot()
    skills = skills if skills is not None else MockSkills()
    obs = obs if obs is not None else MockObs()
    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {}
    try:
        exec(compile(code_str, "<policy_code>", "exec"), safe_globals, safe_locals)  # noqa: S102 - restricted ns
        fn = safe_locals.get("policy_chunk") or safe_locals.get("policy_step")
        if fn is None:
            out["error"] = "no policy_chunk/policy_step defined"
            return out
        # call with whichever signature the fn declares
        import inspect

        nargs = len(inspect.signature(fn).parameters)
        if nargs >= 4:
            result = fn(obs, state, robot, skills)
        else:
            result = fn(obs, robot, skills)
        out["ok"] = True
        out["result"] = result
        out["robot_log"] = robot.log
    except Exception as exc:
        out["error"] = f"exec: {type(exc).__name__}: {exc}"
    return out
