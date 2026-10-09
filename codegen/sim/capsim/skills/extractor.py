"""Regex extraction of top-level function definitions from trial code."""

from __future__ import annotations

import re

# top-level (column-0) ``def`` plus its indented body
FUNCTION_DEF_PATTERN = re.compile(
    r'^(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*\n(?:(?:[ \t]+.+\n?)+))',
    re.MULTILINE,
)

FUNCTION_SIG_PATTERN = re.compile(
    r'^def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:',
    re.MULTILINE,
)

DOCSTRING_PATTERN = re.compile(
    r'^\s+(?:\"\"\"([\s\S]*?)\"\"\"|\'\'\'([\s\S]*?)\'\'\')',
    re.MULTILINE,
)


def extract_docstring(func_code: str) -> str:
    """Return the triple-quoted docstring right after the ``def`` line, or ""."""
    lines = func_code.split("\n", 1)
    if len(lines) < 2:
        return ""
    body = lines[1]
    match = DOCSTRING_PATTERN.match(body)
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    return ""


def extract_functions(code: str) -> list[dict]:
    """Return a dict (name, signature, code, docstring) per top-level function in code."""
    functions: list[dict] = []

    for match in FUNCTION_DEF_PATTERN.finditer(code):
        full_def = match.group(1).rstrip()

        sig_match = FUNCTION_SIG_PATTERN.match(full_def)
        if sig_match:
            name = sig_match.group(1)
            params = sig_match.group(2).strip()
            return_type = sig_match.group(3).strip() if sig_match.group(3) else None

            signature = f"def {name}({params})"
            if return_type:
                signature += f" -> {return_type}"

            docstring = extract_docstring(full_def)

            functions.append({
                "name": name,
                "signature": signature,
                "code": full_def,
                "docstring": docstring,
            })

    return functions
