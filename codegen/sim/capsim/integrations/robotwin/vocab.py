"""The API's data files: object_vocab.yaml (noun-keyed facts) and grounding_policy.yaml (tuning values)."""
from __future__ import annotations

import functools
import re
from pathlib import Path

VOCAB_FILE = Path(__file__).with_name("object_vocab.yaml")


@functools.lru_cache(maxsize=1)
def vocab() -> dict:
    import yaml

    return yaml.safe_load(VOCAB_FILE.read_text())


def words_re(words, flags=re.IGNORECASE) -> re.Pattern:
    """Whole-word match of any entry: ``\\b(w1|w2|...)\\b``."""
    return re.compile(r"\b(" + "|".join(words) + r")\b", flags)


def names_re(names, prefixes, flags=re.IGNORECASE) -> re.Pattern:
    """The whole name is one of ``names``, optionally preceded by the ``prefixes`` in order."""
    pre = "".join(rf"({p}\s+)?" for p in prefixes)
    return re.compile(r"^" + pre + "(" + "|".join(names) + r")$", flags)


def section(name: str) -> dict:
    return vocab()[name]


@functools.lru_cache(maxsize=1)
def policy() -> dict:
    import yaml

    return yaml.safe_load(Path(__file__).with_name("grounding_policy.yaml").read_text())
