"""Repair round: the coder sees its own output + the validation problems."""
from __future__ import annotations

from .client import call_model
from .media import _txt


def repair_code(cmsgs, raw, problems, key, model):
    """One repair round: show the coder its own output + the validation problems and ask
    for a fixed version of the SAME slice (full original context, frames included)."""
    msgs = list(cmsgs) + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": [_txt(
            "Your code failed static validation:\n- " + "\n- ".join(problems) +
            "\nRewrite the COMPLETE code for this slice with these problems fixed, keeping "
            "everything else (calls, order, comments) unchanged. Output only the code.")]},
    ]
    return call_model(msgs, key, model, max_tokens=4096, temperature=0.1)
