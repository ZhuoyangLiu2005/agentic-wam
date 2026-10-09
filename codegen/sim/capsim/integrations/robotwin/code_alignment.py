"""Per-frame code-line alignment for RoboTwin episode collection.

Each executed motion is recorded with the program line that issued it and its frame span in RoboTwin's
``FRAME_IDX`` (1:1 with ``episode{N}.hdf5``); ``alignment/episode{N}.json`` holds seed, success, code and
``records: [{"line", "top_line", "top_locals", "api", "hand", "frame_start", "frame_end"}]``. Uncovered
frames belong to the preceding record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def current_program_line() -> int | None:
    """Innermost line of the exec'd program (``co_filename == "<string>"``) on the stack, or None."""
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code.co_filename == "<string>":
            return int(frame.f_lineno)
        frame = frame.f_back
    return None


_SIMPLE = (int, float, str, bool, type(None))


def _simple_locals(f_locals: dict, limit: int = 64) -> dict:
    """JSON-safe scalar locals, which slicing uses to tell ``[stage each]`` loop iterations apart."""
    out = {}
    for k, v in f_locals.items():
        if k.startswith("_") or len(out) >= limit:
            continue
        if isinstance(v, _SIMPLE):
            out[k] = v
        elif isinstance(v, (tuple, list)) and all(isinstance(x, _SIMPLE) for x in v):
            out[k] = list(v)
    return out


def current_program_top_context() -> tuple[int | None, dict]:
    """(line, simple locals) of the outermost program frame; disambiguates stages that share a helper."""
    frame, top = sys._getframe(), None
    while frame is not None:
        if frame.f_code.co_filename == "<string>":
            top = frame
        frame = frame.f_back
    if top is None:
        return None, {}
    return int(top.f_lineno), _simple_locals(top.f_locals)


def current_program_top_line() -> int | None:
    return current_program_top_context()[0]


def record_motion(env: Any, api_name: str, frame_start: int, frame_end: int, hand: str | None = None) -> None:
    """Append one motion's alignment record (``hand`` = the arm moved) to the env's buffer."""
    buffer = getattr(env, "code_alignment_buffer", None)
    if buffer is None:
        return
    top_line, top_locals = current_program_top_context()
    buffer.append(
        {
            "line": current_program_line(),
            "top_line": top_line,
            "top_locals": top_locals,
            "api": api_name,
            "hand": hand,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
        }
    )


def flush_episode_alignment(
    save_dir: str,
    ep_num: int,
    *,
    seed: int,
    success: bool,
    instruction: str | None,
    num_frames: int,
    code: str | None,
    records: list[dict[str, Any]],
    language: dict[str, Any] | None = None,
) -> str:
    """Write one episode's alignment JSON (``instruction`` stays the unfilled template) and return its path."""
    out_dir = Path(save_dir) / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode{ep_num}.json"
    doc: dict[str, Any] = {
        "seed": int(seed),
        "ep_num": int(ep_num),
        "success": bool(success),
        "instruction": instruction,
        "num_frames": int(num_frames),
        "code": code,
        "records": records,
    }
    if language:
        doc.update({k: language[k] for k in ("placeholders", "instructions", "expert_success", "source", "arm_vars",
                                             "needs_expert", "error", "materialized") if k in language})
    path.write_text(json.dumps(doc, indent=1))
    return str(path)


__all__ = ["current_program_line", "record_motion", "flush_episode_alignment"]
