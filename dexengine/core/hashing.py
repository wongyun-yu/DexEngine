"""File hashing: the one owner of the streamed SHA-256 of a file (identity of checkpoints, archives, inputs)."""

from __future__ import annotations

import hashlib
import os

CHUNK = 1 << 22


def sha256_file(path: str | os.PathLike) -> str:
    """Hex SHA-256 of the file's bytes, read in 4 MiB chunks (the digest does not depend on the chunk size)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()
