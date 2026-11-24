"""Writing base-tree files: JSON written only when its bytes change, and the build location of a tree.

Caches downstream key on file content or (size, mtime); rewriting a tree whose content is unchanged would invalidate
them for nothing. ``write_json_if_changed`` compares the new bytes with the file and writes (temporary file +
``os.replace``) only when they differ. The serialisation is fixed here (``json.dumps(..., ensure_ascii=False,
indent=1)``, UTF-8, no trailing newline) so that every writer produces the same bytes.

A tree is built in ``<tree name>.tmp`` next to its final place and renamed when complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Suffix of the folder a tree is built in (``<tree name><suffix>``).
TREE_TMP_SUFFIX = ".tmp"


def dumps_tree_json(obj) -> bytes:
    """The canonical serialisation of a base-tree JSON file."""
    return json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")


def write_json_if_changed(path, obj) -> bool:
    """Write `obj` to `path` unless the file already has these bytes. True when it wrote."""
    p = Path(path)
    data = dumps_tree_json(obj)
    try:
        if p.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    return True


def tree_tmp(dst):
    """The folder tree `dst` (Path) is built in: same parent, name + TREE_TMP_SUFFIX."""
    return dst.with_name(dst.name + TREE_TMP_SUFFIX)


def tree_tmp_glob(clip: str) -> str:
    """Glob of the build folders of parent clip `clip` (its `_segNN` segments included)."""
    return f"{clip}*{TREE_TMP_SUFFIX}"
