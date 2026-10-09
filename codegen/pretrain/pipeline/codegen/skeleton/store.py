"""The skeleton cache — content-addressed, atomic, and honest about staleness."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def skeleton_key(label: str, prompt_key: str, embodiment: str) -> str:
    """Cache key for one label under one prompt version.

    The PROMPT VERSION is in the key on purpose. The collaborator's version keys
    on the label text alone, so editing a prompt silently keeps serving code
    written by the old one — across the whole corpus at once, with nothing in the
    artifacts to show it. Here a prompt change simply misses, and Stage A refills
    what it needs."""
    h = hashlib.sha256(f"{embodiment}\n{prompt_key}\n{label}".encode()).hexdigest()
    return h[:20]


class SkeletonStore:
    """One JSON per skeleton under ``<root>/<key>.json``.

    Writes are atomic (temp file + ``os.replace``) so any number of generator
    processes can fill the same store concurrently without a lock, which is what
    makes sharded generation (`--shard i/N`) safe."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str):
        p = self.path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None            # a half-written file from a killed run: refill it

    def put(self, key: str, record: dict) -> None:
        p = self.path(key)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(record, indent=1))
        os.replace(tmp, p)         # atomic across processes

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*.json"))

    def stats(self) -> dict:
        n = ok = 0
        lines = []
        for p in self.root.glob("*.json"):
            n += 1
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            ok += 1
            lines.append(len(d.get("code") or []))
        return {"files": n, "readable": ok,
                "mean_lines": round(sum(lines) / max(len(lines), 1), 1),
                "empty": sum(1 for x in lines if x == 0)}
