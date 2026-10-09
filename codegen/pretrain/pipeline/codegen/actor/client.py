"""VLM client for any OpenAI-compatible chat-completions endpoint (VLM_URL) + output fence
stripping."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

from codegen import config

#: chat-completions endpoint of the actor VLM (config key VLM_URL)
VLM_URL = config.get("VLM_URL")

MODEL = config.get("CODER_MODEL")


def model_short(model: str | None = None) -> str:
    """The run-dir path segment for a model id
    ('gcp/google/gemini-3.1-pro-preview' -> 'gemini-3.1-pro').

    ONE definition on purpose: the actor derives its output path from the model
    it actually called, and the batch driver/gate must derive the SAME segment
    or it scores a directory the actor never wrote (symptom: every unit in the
    round comes back `missing run outputs`, and --skip-done re-runs work that
    is already on disk)."""
    m = (model or MODEL) or ""
    return m.rstrip("/").split("/")[-1].replace("-preview", "")

#: upper bound for the finish_reason="length" budget escalation in call_model.
MAX_TOKENS_CEILING = 65536


#: HTTP statuses that will never succeed on retry. Retrying an expired or
#: mistyped key just burns ACTOR_TIMEOUT per unit and fails the whole round with
#: one cryptic line — measured: 4 tries x backoff = ~20 s wasted per episode.
FATAL_STATUS = {400, 401, 403, 404}
#: cap on a single backoff sleep, so a huge Retry-After cannot park a worker
MAX_BACKOFF = 60.0


#: per-process call accounting — appended on every SUCCESSFUL call so an episode
#: can report what it actually cost (wall clock and tokens are the two numbers a
#: campaign is planned from, and neither is recoverable after the fact).
USAGE: list = []

#: every throttled/failed attempt, with whatever the server said about its limits.
#: A 429 from this router arrives with an EMPTY BODY, so without capturing the
#: headers a rate-limit wall teaches nothing and the usable concurrency can only
#: be bisected by burning episodes.
THROTTLE: list = []
#: headers worth keeping: the standard rate-limit families plus Retry-After
_LIMIT_HDR = ("retry-after", "x-ratelimit", "ratelimit", "x-request-id",
              "x-envoy", "retry_after")


class AuthError(RuntimeError):
    """Credential problem — never retried, never re-queued (a retry cannot fix it)."""


def load_key() -> str:
    """The VLM API key: env VLM_API_KEY, else the FIRST non-comment line of a
    `.vlm_api_key` file (workspace root or codegen/pretrain/pipeline/). Only the
    first line: posting a multi-key file as one Bearer token fails with a bare
    401 that looks like an expired key rather than a parse error."""
    import os
    if os.environ.get("VLM_API_KEY"):
        return os.environ["VLM_API_KEY"].strip()
    if not VLM_URL:
        raise RuntimeError("VLM_URL is not configured (vlm_url in configs/<emb>.yaml, or env VLM_URL)")
    for p in (Path(".vlm_api_key"), Path(__file__).resolve().parents[2] / ".vlm_api_key"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    return line
            raise FileNotFoundError(f"{p} is empty")
    raise FileNotFoundError("no VLM key: set VLM_API_KEY or write .vlm_api_key (chmod 600)")


def _backoff(attempt: int, resp=None) -> float:
    """Seconds to wait: the server's own Retry-After if it sent one, else
    exponential, always with jitter.

    The jitter is not decoration. PARALLEL actors share one key, so they hit the
    rate limit together and — with a fixed schedule — would retry together, in
    step, forever. Spreading them is what lets a batch drain instead of
    thundering."""
    import random
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return min(MAX_BACKOFF, float(ra)) + random.uniform(0.0, 1.0)
            except ValueError:
                pass
    return min(MAX_BACKOFF, 2.0 * (2 ** attempt)) + random.uniform(0.0, 2.0)


def call_model(messages, key, model=MODEL, max_tokens=2048, temperature=0.2, retries=6) -> str:
    """Call any model on the NVIDIA OpenAI-compatible router (Gemini / Claude / GPT)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if not model.startswith("openai/"):
        body["temperature"] = temperature
    last = ""
    for i in range(retries):
        t_call = time.time()
        try:
            r = requests.post(
                VLM_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=180,
            )
            if r.status_code == 200:
                choice = r.json()["choices"][0]
                content = choice["message"]["content"]
                reason = choice.get("finish_reason")
                if reason == "length":
                    # TRUNCATED at the output cap. This is retryable whether or not
                    # content came back non-empty: a half-written slice usually still
                    # PARSES (`robot.open_` is a valid attribute expression), so it
                    # would sail through validation and be committed silently, and the
                    # gate pins ZERO events for that slice. Measured on the yam v3
                    # campaign: 21 of 22 mismatched slices pinned nothing, most of them
                    # cut-off generations. Escalate the budget and try again.
                    last = (f"truncated at the output cap (finish_reason=length, "
                            f"{len(content or '')} chars, max_tokens={body['max_tokens']})")
                    body["max_tokens"] = min(body["max_tokens"] * 2, MAX_TOKENS_CEILING)
                elif content:
                    USAGE.append({
                        "prompt_tokens": int((r.json().get("usage") or {}).get("prompt_tokens", 0)),
                        "completion_tokens": int((r.json().get("usage") or {}).get("completion_tokens", 0)),
                        "seconds": round(time.time() - t_call, 3),
                        "model": model,
                    })
                    return content
                else:
                    last = f"empty content (finish_reason={reason}) — reasoning ate max_tokens?"
            elif r.status_code in FATAL_STATUS:
                # an auth/routing error is permanent: fail NOW, loudly, with the
                # fix in the message, instead of retrying a wall
                hint = ("check VLM_API_KEY / .vlm_api_key — the key is missing, malformed or "
                        "EXPIRED (the router says so in the body)"
                        if r.status_code in (401, 403) else
                        f"check the model id {model!r} and the request body")
                raise AuthError(f"model call refused with {r.status_code} "
                                f"({r.text[:200]}) — {hint}")
            else:
                last = f"{r.status_code}: {r.text[:160]}"
                wait = _backoff(i, r)
                THROTTLE.append({
                    "status": r.status_code, "attempt": i + 1, "sleep": round(wait, 1),
                    "headers": {k: v for k, v in r.headers.items()
                                if any(h in k.lower() for h in _LIMIT_HDR)},
                    "body": (r.text or "")[:120]})
                if r.status_code == 429 and i == 0:
                    print(f"[client] 429 from the router — headers: "
                          f"{THROTTLE[-1]['headers'] or '(none: the limit is not advertised)'}"
                          f"  body={THROTTLE[-1]['body']!r}", file=sys.stderr)
                time.sleep(wait)               # honours Retry-After on 429/5xx
                continue
        except AuthError:
            raise
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(_backoff(i))
    raise RuntimeError(f"model call ({model}) failed after {retries} tries: {last}")


def _strip_fences(code: str):
    out = []
    for ln in code.splitlines():
        if ln.strip().startswith("```"):
            continue
        out.append(ln.rstrip())
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return [ln for ln in out if ln.strip()]
