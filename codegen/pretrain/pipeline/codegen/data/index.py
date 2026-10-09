"""Corpus work-unit enumeration — the dispatch layer's view of the dataset.

A WORK UNIT is one (task, episode_index) pair: the thing one actor annotates,
one manifest row records, and one frontier line names (rendered ``task`` when
the index is 0, ``task:idx`` otherwise — the format ``run_batch --tasks-file``
consumes).

This module answers "what exists" from dataset METADATA only (info.json /
directory listings) — it never opens a parquet or a video, so it stays cheap
enough to call per dispatch. At full-corpus scale this is the seam where a
prebuilt index file replaces live enumeration: keep the return shape, change
the source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _selection_units(path: Path) -> list[tuple[str, int]]:
    """Work units from a SELECTION FILE — the campaign's chosen subset.

    Two accepted forms:
      * JSON: a list (or {"episodes": [...]}) of entries carrying ``subset``
        (the LeRobot dataset dir) and ``episode_index``;
      * a plain text file of ``<dataset>[:<episode>]`` lines (``#`` comments ok),
        i.e. exactly the format ``run_batch --tasks-file`` consumes."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        eps = data.get("episodes", data) if isinstance(data, dict) else data
        pairs = {(str(e["subset"]), int(e["episode_index"])) for e in eps}
    else:
        pairs = set()
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            task, _, idx = line.partition(":")
            pairs.add((task.strip(), int(idx) if idx.strip() else 0))
    return sorted(pairs)


def _has_camera(root: Path, task: str, cam: str) -> bool:
    """Does this dataset ship the camera stream the actor labels from? Checked against the file
    the reader would open (episode 0, through the dataset's own ``video_path`` template)."""
    try:
        info = json.loads((root / task / "meta" / "info.json").read_text())
    except Exception:
        return False
    tmpl = str(info.get("video_path")
               or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return (root / task / tmpl.format(episode_chunk=0, video_key=cam, episode_index=0)).exists()


def corpus_units(embodiment: str) -> list[tuple[str, int]]:
    """Ordered (dataset, episode_index) pairs of a pretraining source (yam | ego)."""
    from codegen import config
    from codegen.data.readers import SOURCES, camera, list_datasets, source_root
    if embodiment not in SOURCES:          # post-training embodiments: one unit per spec task
        from codegen.embodiments import get_spec
        return [(task, 0) for task in get_spec(embodiment).list_tasks()]
    root, cam = source_root(embodiment), camera(embodiment)
    sel = config.get("SELECTION")
    if sel:
        units = _selection_units(Path(sel) if Path(sel).is_absolute() else config.ROOT / sel)
        bad = sorted({t for (t, _i) in units if not _has_camera(root, t, cam)})
        if bad:
            units = [(t, i) for (t, i) in units if t not in bad]
            print(f"[index] WARNING selection dropped {len(bad)} dataset(s) with no "
                  f"'{cam}' stream: {bad[:3]}{'...' if len(bad) > 3 else ''}", file=sys.stderr)
        print(f"[index] selection {sel}: {len(units)} work units "
              f"({len({t for t, _ in units})} datasets)", file=sys.stderr)
        return units
    units, skipped = [], []
    for task in list_datasets(embodiment):
        if not _has_camera(root, task, cam):
            skipped.append(task)
            continue
        info = json.loads((root / task / "meta" / "info.json").read_text())
        units += [(task, i) for i in range(int(info.get("total_episodes", 1)))]
    if skipped:
        # loud, never a silent shrink: these are episodes the campaign CANNOT label
        print(f"[index] {len(skipped)} dataset(s) excluded: no '{cam}' stream — "
              f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}", file=sys.stderr)
    return units


def unit_name(task: str, episode_index: int) -> str:
    return task if episode_index == 0 else f"{task}:{episode_index}"
