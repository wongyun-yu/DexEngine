"""Repo-relative path references in text files (part of the scan in tests/core/references_scan.py)."""

from __future__ import annotations

import os
import re

from dexengine.core.artifact_paths import OPS_DIR

#: The deployment layer's top folder (its one owner is artifact_paths.OPS_DIR; a tree may have none).
SITE = os.path.basename(OPS_DIR)
TEXT_EXT = (".py", ".sh", ".yaml", ".yml", ".md", ".toml", ".txt", ".env")
PATH_RE = re.compile(
    r"(?<![\w/.$-])((?:"
    + "|".join(("dexengine", "experiments", SITE, "scripts", "configs", "tests", "assets"))
    + r")/[\w./-]*\w\."
    r"(?:py|sh|yaml|yml|json|tsv|txt|md|env))(?!\w)"  # whole extension (a .jsonl file is not a .json path)
)


def path_refs(text: str) -> list[tuple[int, str]]:
    """(line, path) for every repo-relative path in `text`; patterns (`*`, `{}`, `<>`) are not paths."""
    return [
        (i, p)
        for i, line in enumerate(text.splitlines(), 1)
        for p in PATH_RE.findall(line)
        if not any(c in p for c in "*{<")
    ]


def missing_paths(texts: dict[str, str], present: set[str]) -> list[tuple[str, int, str]]:
    """(file, line, path) for every quoted path that is not in `present` (files and their parent directories)."""
    out = []
    for rel, text in texts.items():
        if rel.endswith(TEXT_EXT):
            out += [(rel, i, p) for i, p in path_refs(text) if p.rstrip("/") not in present]
    return out


def with_parents(files: set[str]) -> set[str]:
    out = set(files)
    for f in files:
        parts = f.split("/")
        out.update("/".join(parts[:k]) for k in range(1, len(parts)))
    return out
