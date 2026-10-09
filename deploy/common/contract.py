#!/usr/bin/env python
"""Wire contract shared by BOTH deploy lines (local GPU-on-robot, remote GPU-on-cluster).

Stdlib only, no numpy/torch — so the robot host, the GPU host, and the link-qualification tools
can all import it without dragging in the model stack.

Why a versioned contract at all: over loopback a schema mismatch shows up instantly as a crash.
Over a network it shows up as a robot executing a chunk built from a misread state vector. Every
field is validated on arrival and PROTOCOL_VERSION is checked on every request, so a client/server
skew fails loudly at the first call instead of silently at the joints.

  /act      ActRequest  -> ActResponse
  /reset    {}          -> {ok, episode}
  /health   -            -> HealthResponse-ish dict (informational, not validated)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = 1

# Camera keys as the policy expects them. head_camera == cam_high (top-down),
# left/right == the wrist cams. Order is NOT significant; presence is.
CAMERAS = ("head_camera", "left_camera", "right_camera")

IMAGE_FORMATS = ("jpeg", "png")

# The server returns ABSOLUTE joint targets: the model predicts deltas via RelativeJointTransform
# and the server adds the request's state back before replying.
ACTION_TYPE_ABS = "qpos_abs"


class ContractError(ValueError):
    """Malformed request/response. Always carries a message naming the offending field."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


@dataclass
class ActRequest:
    """Robot -> server. One observation, one replan."""
    instruction: str
    images: Dict[str, Dict[str, str]]      # {camera: {"format": "jpeg"|"png", "data": <base64>}}
    state: List[float]                     # raw joint vector, [Ljoint0..5, Lgrip, Rjoint0..5, Rgrip]
    seq: int = 0                           # client-side monotonic counter, echoed back
    obs_stamp_ns: int = 0                  # when the observation was captured (client clock)
    protocol: int = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {"protocol": self.protocol, "instruction": self.instruction, "images": self.images,
                "state": list(self.state), "seq": self.seq, "obs_stamp_ns": self.obs_stamp_ns}

    @staticmethod
    def from_dict(d: Dict[str, Any], expect_state_dim: Optional[int] = None) -> "ActRequest":
        _require(isinstance(d, dict), "body must be a JSON object")
        proto = d.get("protocol", PROTOCOL_VERSION)
        _require(proto == PROTOCOL_VERSION,
                 f"protocol mismatch: client sent {proto}, server speaks {PROTOCOL_VERSION}")

        imgs = d.get("images")
        _require(isinstance(imgs, dict), "images: expected an object keyed by camera name")
        for cam in CAMERAS:
            _require(cam in imgs, f"images: missing camera '{cam}' (need all of {list(CAMERAS)})")
            entry = imgs[cam]
            _require(isinstance(entry, dict), f"images['{cam}']: expected an object")
            fmt = entry.get("format", "png")
            _require(fmt in IMAGE_FORMATS, f"images['{cam}'].format: '{fmt}' not in {list(IMAGE_FORMATS)}")
            _require(isinstance(entry.get("data"), str) and entry["data"],
                     f"images['{cam}'].data: expected a non-empty base64 string")

        st = d.get("state")
        _require(isinstance(st, (list, tuple)) and len(st) > 0, "state: expected a non-empty list")
        try:
            st = [float(x) for x in st]
        except (TypeError, ValueError) as e:
            raise ContractError(f"state: non-numeric entry ({e})")
        _require(all(x == x for x in st), "state: contains NaN")   # NaN != NaN
        if expect_state_dim is not None:
            _require(len(st) == expect_state_dim,
                     f"state: expected {expect_state_dim} values, got {len(st)}")

        instr = d.get("instruction")
        _require(instr is None or isinstance(instr, str), "instruction: expected a string")

        return ActRequest(instruction=instr or "", images=imgs, state=st,
                          seq=int(d.get("seq", 0)), obs_stamp_ns=int(d.get("obs_stamp_ns", 0)),
                          protocol=proto)


@dataclass
class ActResponse:
    """Server -> robot. One chunk of ABSOLUTE joint targets, plus timing telemetry.

    `timing` is what lets the client separate network cost from compute cost without guessing:
    round_trip - infer_ms == everything the link and serialization cost.
    """
    action_chunk: List[List[float]]
    horizon: int
    action_dim: int
    replan_steps: int
    action_type: str = ACTION_TYPE_ABS
    instruction: str = ""
    code: Optional[str] = None             # generated policy code (3-expert); None for 2-expert
    seq: int = 0                           # echoed from the request
    obs_stamp_ns: int = 0                  # echoed, so the client can age the chunk
    timing: Dict[str, float] = field(default_factory=dict)
    protocol: int = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {"protocol": self.protocol, "action_chunk": self.action_chunk,
                "horizon": self.horizon, "action_dim": self.action_dim,
                "replan_steps": self.replan_steps, "action_type": self.action_type,
                "instruction": self.instruction, "code": self.code,
                "seq": self.seq, "obs_stamp_ns": self.obs_stamp_ns, "timing": self.timing}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ActResponse":
        _require(isinstance(d, dict), "response must be a JSON object")
        proto = d.get("protocol", PROTOCOL_VERSION)
        _require(proto == PROTOCOL_VERSION,
                 f"protocol mismatch: server sent {proto}, client speaks {PROTOCOL_VERSION}")

        chunk = d.get("action_chunk")
        _require(isinstance(chunk, list) and len(chunk) > 0, "action_chunk: expected a non-empty list")
        dim = int(d.get("action_dim", len(chunk[0]) if chunk and isinstance(chunk[0], list) else 0))
        for i, row in enumerate(chunk):
            _require(isinstance(row, list), f"action_chunk[{i}]: expected a list")
            _require(len(row) == dim, f"action_chunk[{i}]: expected {dim} values, got {len(row)}")
            for v in row:
                _require(isinstance(v, (int, float)) and v == v,
                         f"action_chunk[{i}]: non-finite or non-numeric value")

        _require(d.get("action_type") == ACTION_TYPE_ABS,
                 f"action_type: expected '{ACTION_TYPE_ABS}', got {d.get('action_type')!r} "
                 f"(a relative chunk commanded as absolute would drive the arm to the wrong place)")

        return ActResponse(action_chunk=chunk, horizon=int(d.get("horizon", len(chunk))),
                           action_dim=dim, replan_steps=int(d.get("replan_steps", len(chunk))),
                           action_type=d["action_type"], instruction=d.get("instruction", ""),
                           code=d.get("code"), seq=int(d.get("seq", 0)),
                           obs_stamp_ns=int(d.get("obs_stamp_ns", 0)),
                           timing=d.get("timing") or {}, protocol=proto)
