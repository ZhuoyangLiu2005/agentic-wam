"""Bound per-sample loading latency so one slow read cannot stall the whole job.

PyTorch's DataLoader delivers batches STRICTLY IN ORDER: if the worker owning the next batch is
slow, batches already finished by other workers cannot be handed over, so the rank blocks. The
ZeRO all-reduce then amplifies that 48x -- every other rank spins in NCCL waiting for it (which
reads as 100% GPU util, not idle). One tail-latency read therefore idles the entire cluster, and
if it exceeds the watchdog timeout it kills the run.

Storage tail latency is the trigger here: yam samples open 3 video files each from Lustre, and with
48 ranks x N workers the p99.9 is seconds even when the median is ~5 ms.

This wrapper puts a deadline on each __getitem__ (SIGALRM, which interrupts blocking syscalls).
On timeout it retries a different random index, then falls back to the last good sample. Over
~3e8 samples, substituting a tiny fraction is statistically irrelevant; stalling 48 GPUs is not.

Counters are exposed for monitoring: `n_timeout`, `n_error`, `n_fallback`.
"""
from __future__ import annotations

import logging
import os
import random
import signal

import torch

logger = logging.getLogger(__name__)


class _Timeout(Exception):
    pass


# A SIGALRM can be delivered AFTER we disarm the timer (the signal is already pending). If the
# handler raised unconditionally it would land inside the DataLoader worker's IPC code, outside
# our try/except -- silently killing the worker so its batch never arrives and the parent blocks
# in _try_get_data forever, with no timeout logged; a multi-node run stalled exactly this way.
# The gate makes any late signal inert: only raise while we are inside __getitem__.
_ACTIVE = [False]


def _on_alarm(signum, frame):        # noqa: ARG001
    if _ACTIVE[0]:
        raise _Timeout()


class DeadlineDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, seconds: float = 30.0, max_retries: int = 2,
                 log_every: int = 50):
        self.ds = dataset
        self.seconds = float(os.environ.get("AGENTICWAM_SAMPLE_DEADLINE_S", seconds))
        self.enabled = self.seconds > 0      # AGENTICWAM_SAMPLE_DEADLINE_S=0 -> plain passthrough
        self.max_retries = int(max_retries)
        self.log_every = int(log_every)
        self._last = None
        self.n_timeout = 0
        self.n_error = 0
        self.n_fallback = 0

    def __len__(self):
        return len(self.ds)

    @property
    def datasets(self):                      # keep ConcatDataset introspection working
        return getattr(self.ds, "datasets", [self.ds])

    def __getitem__(self, i):
        if not self.enabled:
            return self.ds[i]
        n = len(self.ds)
        for k in range(self.max_retries + 1):
            idx = i if k == 0 else random.randrange(n)
            prev = None
            try:
                prev = signal.signal(signal.SIGALRM, _on_alarm)
                _ACTIVE[0] = True
                signal.setitimer(signal.ITIMER_REAL, self.seconds)
            except ValueError:               # not the main thread of this process -> no deadline
                try:
                    s = self.ds[idx]; self._last = s; return s
                except Exception:            # noqa: BLE001
                    self.n_error += 1
                    if self._last is not None:
                        self.n_fallback += 1; return self._last
                    raise
            try:
                s = self.ds[idx]
                self._last = s
                return s
            except _Timeout:
                self.n_timeout += 1
                if (self.n_timeout % self.log_every) == 1:
                    logger.warning("sample deadline %.0fs exceeded (idx=%s) timeouts=%d errors=%d "
                                   "fallbacks=%d", self.seconds, idx, self.n_timeout,
                                   self.n_error, self.n_fallback)
            except Exception:                # noqa: BLE001  corrupt/missing sample -> substitute too
                self.n_error += 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                _ACTIVE[0] = False           # disarm BEFORE restoring: late signals become no-ops
                if prev is not None:
                    signal.signal(signal.SIGALRM, prev)
        if self._last is not None:
            self.n_fallback += 1
            return self._last
        return self.ds[i]                    # nothing cached yet: block rather than fail the run
