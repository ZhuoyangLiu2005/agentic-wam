"""Forward-only lazy frame access — the episode's clip WITHOUT materializing it.

The pipeline decodes a whole episode into ``(T,H,W,3)`` uint8 but indexes only
a handful of frames (endpoints per slice + one pre-frame per contact event):
typically a few percent of an episode's frames. At
corpus scale that resident array is the binding constraint — ~2.4 GB per
episode of 2,828 frames, times one per parallel actor.

``LazyClip`` keeps the SAME ``clip[i]`` / ``len(clip)`` interface but holds at
most one frame at a time. Two facts make it work:

  * H.264 is inter-frame compressed, so every frame must pass through the
    decoder — but ``VideoCapture.grab()`` decodes WITHOUT the colour conversion
    and allocation that ``retrieve()`` does. Skipping that for the ~97% of
    frames nobody asks for is where the saving comes from (measured on a yam
    episode: 460 MB -> 12 MB resident, and 0.71 s -> 0.13 s as a side effect).
  * The pipeline's access is strictly monotonically increasing (verified on
    both embodiments), so ONE
    forward pass serves every request.

Random seeking is the wrong shape and measurably slower, not faster: producing
frame N means decoding from the preceding keyframe forward, so seeking repeats
that work per target (measured 4x slower than this accessor on a 499-frame
episode; ~14x on a 2,828-frame one).

A backward request is still CORRECT — the capture reopens and re-walks — but
it is counted and warned about once, so the monotonicity assumption failing
becomes visible instead of quietly slow.

``first_frame``/``length`` window a larger file: this is the packed-shard case
(several episodes per mp4; an episode is a frame range located by its meta row). A shard reader only needs to resolve the window and hand it
to ``LazyClip(shard_mp4, first_frame=..., length=...)`` — the pipeline sees an
ordinary episode clip.
"""
from __future__ import annotations

import sys
from pathlib import Path


class LazyClip:
    """Frames of ONE video, decoded forward on demand.

    Supports what the pipeline uses: ``len(clip)``, ``clip[i]`` -> (H,W,3)
    uint8 RGB, and ``clip[:n]`` -> a length-truncated view (the readers' way of
    aligning a clip to a shorter state table)."""

    def __init__(self, path: Path, length: int | None = None, first_frame: int = 0):
        """``first_frame``/``length`` select a WINDOW of a larger file — the
        shape of a packed multi-episode shard, where one episode is a frame
        range inside a ~114-episode mp4. ``clip[0]`` maps to global frame
        ``first_frame``; the capture seeks there ONCE per open (a single seek
        per episode amortizes; it is per-target repeated seeking that is
        slow), then walks forward as usual."""
        self.path = Path(path)
        self._first = int(first_frame)
        self._cap = None
        self._next = 0          # LOCAL index the decoder is positioned to grab next
        self._cache: tuple[int, object] | None = None
        self.rewinds = 0
        if length is not None:
            self._len = int(length)
        else:
            self._len = max(0, self._probe_length() - self._first)

    # ------------------------------------------------------------- internals
    def _probe_length(self) -> int:
        import cv2
        cap = cv2.VideoCapture(str(self.path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
        cap = cv2.VideoCapture(str(self.path))     # unreliable header: count
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        if n == 0:
            import re
            m = re.search(r"FFMPEG:\s*(\S+)", cv2.getBuildInformation())
            if m and m.group(1).upper() != "YES":
                raise RuntimeError(
                    f"no frames decoded from {self.path}: this cv2 ({cv2.__version__}) "
                    f"has no FFMPEG backend (FFMPEG: {m.group(1)}) and cannot open "
                    "mp4s. Install opencv-python-headless in an ISOLATED venv — a "
                    "venv created with --system-site-packages still resolves to the "
                    "system cv2. See requirements.txt.")
            raise RuntimeError(f"no frames decoded from {self.path}")
        return n

    def _open(self):
        import cv2
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")
        if self._first:
            # one seek to the window start; cv2+FFMPEG lands this frame-accurately
            # (decodes forward from the preceding keyframe internally) — verified
            # pixel-exact against ground-truth boundaries in the shard fixture test
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._first)
        self._next = 0

    def _read_at(self, i: int):
        """Advance the decoder to frame ``i`` and return it (RGB ndarray)."""
        import cv2
        import numpy as np  # noqa: F401  (returned arrays are numpy)
        if self._cap is None:
            self._open()
        if i < self._next - 1:                      # backward: must re-walk
            self.rewinds += 1
            if self.rewinds == 1:
                print(f"[frames] WARNING {self.path.name}: backward frame access "
                      f"({i} after {self._next - 1}) — reopening and re-decoding. "
                      f"Forward-only access is what makes this cheap; repeated "
                      f"rewinds mean the caller's order is no longer monotonic.",
                      file=sys.stderr)
            self._open()
        while self._next <= i:
            ok = self._cap.grab()                   # decode, no convert/alloc
            if not ok:                              # past the real end: clamp
                if self._cache is not None:
                    return self._cache[1]
                raise RuntimeError(f"no frame {i} in {self.path}")
            self._next += 1
            if self._next - 1 == i:
                ok, bgr = self._cap.retrieve()      # convert ONLY what is used
                if not ok:
                    raise RuntimeError(f"cannot retrieve frame {i} from {self.path}")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise AssertionError("unreachable")

    # --------------------------------------------------------------- mapping
    def __len__(self) -> int:
        return self._len

    def __getitem__(self, k):
        if isinstance(k, slice):
            if k.start not in (None, 0) or k.step not in (None, 1):
                raise TypeError("LazyClip supports only clip[:n] truncation")
            n = self._len if k.stop is None else min(int(k.stop), self._len)
            return LazyClip(self.path, length=n, first_frame=self._first)
        i = int(k)
        if i < 0:
            i += self._len
        if not (0 <= i < self._len):
            raise IndexError(f"frame {k} out of range (len {self._len})")
        if self._cache is not None and self._cache[0] == i:
            return self._cache[1]                   # repeat access is free
        frame = self._read_at(i)
        self._cache = (i, frame)
        return frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
