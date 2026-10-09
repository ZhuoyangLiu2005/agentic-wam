"""Shared, memory-mapped store for the Cosmos-Reason1 prompt embeddings.

WHY: every __getitem__ used to torch.load a 25.7 MB (128, 100352) bf16 tensor from Lustre -- 36 ms
and 25.7 MB of I/O per sample, for only 607 distinct prompts (15.6 GB total). That was ~47x the
video bytes per sample and it evicted the video pages from page cache, which is what made the
48-GPU run degrade from 0.192 to 0.104 step/s as sampling spread over the corpus.

NOW: one consolidated .npy, staged once per node (ideally /dev/shm), mmap'd read-only and shared by
every dataloader worker on that node. Lookup is a zero-copy row slice: ~0.02 ms vs 36 ms.

Enable by setting AGENTICWAM_PROMPT_EMBS_NPY (the .npy; its .json index sits alongside). Unset -> the
per-file torch.load path is used unchanged, so this is opt-in and reversible.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import numpy as np
import torch

_LOCK = threading.Lock()
_STORE = None          # (memmap, {sha256: row})
_TRIED = False


def _load():
    global _STORE, _TRIED
    if _STORE is not None or _TRIED:
        return _STORE
    with _LOCK:
        if _STORE is not None or _TRIED:
            return _STORE
        _TRIED = True
        p = os.environ.get("AGENTICWAM_PROMPT_EMBS_NPY")
        if not p or not os.path.exists(p):
            return None
        idx_p = os.path.splitext(p)[0] + ".json"
        if not os.path.exists(idx_p):
            return None
        arr = np.load(p, mmap_mode="r")
        idx = json.load(open(idx_p))["sha256_to_row"]
        _STORE = (arr, idx)
        return _STORE


def get(prompt: str):
    """-> bf16 tensor [128, D] for `prompt`, or None if the store is unavailable/missing the key."""
    st = _load()
    if st is None:
        return None
    arr, idx = st
    row = idx.get(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    if row is None:
        return None
    # rows are contiguous -> this is a view, no 25.7 MB copy; the collator copies once per batch
    return torch.from_numpy(np.asarray(arr[row])).view(torch.bfloat16)
