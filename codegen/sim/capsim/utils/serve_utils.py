"""Small HTTP helpers for talking to the perception servers."""
import time

import requests


def post_with_retries(
    url: str,
    payload: dict,
    timeout_seconds: float = 120.0,
    retry_interval: float = 1.0,
    max_retries: int = 5,
):
    """POST with exponential backoff (capped at 8 s) until timeout_seconds or max_retries; raises RuntimeError."""
    deadline = time.time() + timeout_seconds
    current_interval = retry_interval

    last_err = None
    attempts = 0
    while time.time() < deadline and attempts < max_retries:
        try:
            resp = requests.post(url, json=payload, timeout=timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            attempts += 1
            time.sleep(min(current_interval, max(0, deadline - time.time())))
            current_interval = min(current_interval * 2, 8.0)

    raise RuntimeError(
        f"Request to {url} failed after {attempts} retries / "
        f"{timeout_seconds:.2f}s. Last error: {last_err}"
    )


def post_with_queue_tolerance(
    url: str,
    payload: dict,
    timeout_seconds: float = 120.0,
    retry_interval: float = 1.0,
    max_retries: int = 5,
):
    """Like post_with_retries, but also retries HTTP 503 (queued/busy server) instead of raising."""
    deadline = time.time() + timeout_seconds
    current_interval = retry_interval

    last_err = None
    attempts = 0
    while time.time() < deadline and attempts < max_retries:
        try:
            resp = requests.post(url, json=payload, timeout=timeout_seconds)
            if resp.status_code == 503:
                last_err = requests.HTTPError(
                    f"503 Service Unavailable: {resp.text}", response=resp
                )
                attempts += 1
                time.sleep(min(current_interval, max(0, deadline - time.time())))
                current_interval = min(current_interval * 2, 8.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            attempts += 1
            time.sleep(min(current_interval, max(0, deadline - time.time())))
            current_interval = min(current_interval * 2, 8.0)

    raise RuntimeError(
        f"Request to {url} failed after {attempts} retries / "
        f"{timeout_seconds:.2f}s. Last error: {last_err}"
    )
