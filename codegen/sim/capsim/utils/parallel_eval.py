"""Run trials across worker processes with a per-worker setup hook."""
from __future__ import annotations

import math
import multiprocessing
import queue as _queue
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

BatchResult = TypeVar("BatchResult")


def chunk_into_batches(items: Sequence[int], batch_size: int) -> list[list[int]]:
    """Split ``items`` into ``batch_size`` chunks."""

    if batch_size <= 0:
        return [list(items)]
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def run_parallel_batches(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    batch_fn: Callable[[Sequence[int]], Iterable[BatchResult]],
    batch_size: int | None = None,
    mp_start_method: str = "spawn",
) -> list[BatchResult]:
    """Run ``batch_fn`` over even batches of ``trial_ids`` in a spawn process pool (sequential if num_workers <= 1); results concatenated."""

    if not trial_ids:
        return []

    if num_workers <= 1 or len(trial_ids) == 1:
        return list(batch_fn(trial_ids))

    if batch_size is None:
        batch_size = max(1, math.ceil(len(trial_ids) / num_workers))

    batches = chunk_into_batches(trial_ids, batch_size)
    ctx = multiprocessing.get_context(mp_start_method)

    results: list[BatchResult] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [executor.submit(batch_fn, batch) for batch in batches]
        for future in as_completed(futures):
            results.extend(future.result())

    return results


def _worker_loop(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    single_trial_fn: Callable[[int], BatchResult],
) -> None:
    """Pull trial ids from task_queue until a None sentinel or an empty queue; report results on result_queue."""
    while True:
        try:
            trial_id = task_queue.get(timeout=1.0)
        except Exception:
            break

        if trial_id is None:
            break

        try:
            result = single_trial_fn(trial_id)
            result_queue.put(("success", trial_id, result))
        except Exception as e:
            result_queue.put(("error", trial_id, str(e)))


def _worker_loop_with_setup(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    setup_fn: Callable[[], Any],
    trial_fn: Callable[[Any, int], BatchResult],
) -> None:
    """Like _worker_loop, but builds worker state once with setup_fn and calls trial_fn(state, trial_id)."""
    try:
        state = setup_fn()
    except Exception as e:
        result_queue.put(("setup_error", worker_id, str(e)))
        return

    while True:
        try:
            trial_id = task_queue.get(timeout=1.0)
        except Exception:
            break

        if trial_id is None:
            break

        try:
            result = trial_fn(state, trial_id)
            result_queue.put(("success", trial_id, result))
        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            result_queue.put(("error", trial_id, error_msg))


def run_parallel_dynamic(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    single_trial_fn: Callable[[int], BatchResult],
    mp_start_method: str = "spawn",
) -> list[BatchResult]:
    """Run trials over a shared work queue (load-balances uneven trial times); result order may differ from input."""
    if not trial_ids:
        return []

    if num_workers <= 1:
        return [single_trial_fn(tid) for tid in trial_ids]

    ctx = multiprocessing.get_context(mp_start_method)
    task_queue: multiprocessing.Queue = ctx.Queue()
    result_queue: multiprocessing.Queue = ctx.Queue()

    for trial_id in trial_ids:
        task_queue.put(trial_id)

    for _ in range(num_workers):
        task_queue.put(None)

    workers = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=_worker_loop,
            args=(worker_id, task_queue, result_queue, single_trial_fn),
        )
        p.start()
        workers.append(p)

    results: list[BatchResult] = []
    errors: list[tuple[int, str]] = []
    for _ in range(len(trial_ids)):
        status, trial_id, data = result_queue.get()
        if status == "success":
            results.append(data)
        else:
            errors.append((trial_id, data))
            print(f"Trial {trial_id} failed with error: {data}")

    for p in workers:
        p.join()

    if errors:
        print(f"WARNING: {len(errors)} trial(s) failed")

    return results


def run_parallel_with_setup(
    trial_ids: Sequence[int],
    *,
    num_workers: int,
    setup_fn: Callable[[], Any],
    trial_fn: Callable[[Any, int], BatchResult],
    mp_start_method: str = "spawn",
    result_watchdog_s: float | None = None,
) -> list[BatchResult]:
    """Run trials over a shared queue with per-worker state (e.g. one sim env per worker).

    Args:
        trial_ids: trial identifiers; num_workers <= 1 runs sequentially.
        setup_fn: picklable; called once per worker to create its state.
        trial_fn: picklable; ``trial_fn(state, trial_id)`` -> result.
        result_watchdog_s: if no result arrives within this time, kill stuck workers and fail the rest.

    Returns:
        Successful results (order may differ from input).
    """
    if not trial_ids:
        return []

    if num_workers <= 1:
        state = setup_fn()
        return [trial_fn(state, tid) for tid in trial_ids]

    ctx = multiprocessing.get_context(mp_start_method)
    task_queue: multiprocessing.Queue = ctx.Queue()
    result_queue: multiprocessing.Queue = ctx.Queue()

    for trial_id in trial_ids:
        task_queue.put(trial_id)

    for _ in range(num_workers):
        task_queue.put(None)

    workers = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=_worker_loop_with_setup,
            args=(worker_id, task_queue, result_queue, setup_fn, trial_fn),
        )
        p.start()
        workers.append(p)

    results: list[BatchResult] = []
    errors: list[tuple[int, str]] = []
    setup_errors = 0
    accounted_ids: set[int] = set()

    try:
        for _ in range(len(trial_ids)):
            try:
                status, identifier, data = result_queue.get(timeout=result_watchdog_s)
            except _queue.Empty:
                # a worker was SIGKILLed (e.g. infinite loop in generated code) and never reported
                alive = [p for p in workers if p.is_alive()]
                print(
                    f"WARNING: result watchdog triggered ({result_watchdog_s}s). "
                    f"Killing {len(alive)} stuck worker(s)."
                )
                for p in alive:
                    p.kill()
                    p.join(timeout=5.0)
                missing = [t for t in trial_ids if t not in accounted_ids]
                for mid in missing:
                    errors.append((mid, "Worker killed by watchdog (infinite loop in generated code)"))
                break

            accounted_ids.add(identifier)
            if status == "success":
                results.append(data)
            elif status == "setup_error":
                setup_errors += 1
                print(f"Worker {identifier} failed during setup: {data}")
            else:
                errors.append((identifier, data))
                print(f"Trial {identifier} failed with error: {data}")
    finally:
        # always clean up workers, including on KeyboardInterrupt
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3.0)
                if p.is_alive():
                    p.kill()  # SIGKILL for workers stuck in C extension code

    if setup_errors:
        print(f"WARNING: {setup_errors} worker(s) failed during setup")
    if errors:
        print(f"WARNING: {len(errors)} trial(s) failed")

    return results


__all__ = [
    "chunk_into_batches",
    "run_parallel_batches",
    "run_parallel_dynamic",
    "run_parallel_with_setup",
]
