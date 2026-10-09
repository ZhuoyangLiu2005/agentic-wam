"""The run-dir artifact SCHEMA — single source of truth for the names every
producer (pipeline.episode) writes and every consumer (pipeline.gate, the viz
server, the coordinator's audits, check_skill evidence) reads.

The schema is EMBODIMENT-INVARIANT on purpose: gate/viz/audits stay
embodiment-blind precisely because every run dir looks the same. Embodiment
differences live in the run-dir PATH (codegen.embodiments spec.run_dir) and
the data source (codegen.data.readers) — never here. An embodiment needing an
extra diagnostic artifact adds an OPTIONAL member here; the required core
never forks.

Episode identity lives in the run-dir path (…/<episode_id>/<tag>/), so
artifact FILENAMES don't carry it. Some existing runs use
``episode_000_``-prefixed names — the ``*_path`` helpers fall back to those so
they stay readable.
"""
from __future__ import annotations

from pathlib import Path

# required core, written by every episode run
PROGRAM = "program.py"          # the assembled, always-compiling program
LINES = "lines.json"            # frame_src_block + grip_pins (frame -> line alignment)
CHUNKS = "chunks.json"          # instruction + per-slice phases
TRACE = "trace.json"            # per-slice VLM I/O, validation, grip_reconcile
CACHE = "cache.json"            # content-fingerprinted model responses
TRACE_DIR = "trace"             # sampled input frames (PNG)

# older artifact names (read-side fallback only; never written)
_LEGACY_LINES = "episode_000_lines.json"
_LEGACY_CHUNKS = "episode_000_chunks.json"


def lines_path(rd: Path) -> Path:
    p = rd / LINES
    return p if p.exists() else rd / _LEGACY_LINES


def chunks_path(rd: Path) -> Path:
    p = rd / CHUNKS
    return p if p.exists() else rd / _LEGACY_CHUNKS
