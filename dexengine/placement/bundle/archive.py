"""The archive of an RL bundle (rl_bundle.py): tar | zstd | split parts, read back as one stream, and file hashes —
plus the bundle tool's failure exit (`die`)."""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from pathlib import Path

from dexengine.core.errors import input_missing

#: Archive part base name (the README tells how to join the parts).
ARCHIVE = "data.tar.zst"
ZSTD_LEVEL = 3  # npz members are already deflated — higher levels buy little
CHUNK = 1 << 22
TAG = "[RL_BUNDLE]"


def die(msg: str, items=()) -> None:
    """Exit with `[RL_BUNDLE] FAIL <msg>` and the first 20 items."""
    items = list(items)
    tail = "".join(f"\n  {x}" for x in items[:20]) + (f"\n  … and {len(items) - 20} more" if len(items) > 20 else "")
    raise SystemExit(f"{TAG} FAIL {msg}{tail}")


def sh(p) -> str:
    """Single-quote a path for bash."""
    return "'" + str(p).replace("'", "'\\''") + "'"


def pack(root: Path, rels: list[str], out: Path, part_size: str) -> list[Path]:
    """tar (relative paths · NUL list · symlinks and hard links dereferenced — every member a regular file, as
    read_archive requires) | zstd | split. Every rc is checked (pipefail) — tar rc 1 (file changed) fails too."""
    lst = out / ".files0"
    lst.write_bytes(b"".join(r.encode() + b"\0" for r in rels))
    cmd = (
        f"set -o pipefail; tar -C {sh(root)} --null --verbatim-files-from -T {sh(lst)} --dereference "
        f"--hard-dereference --owner=0 --group=0 --numeric-owner -cf - | zstd -T0 -{ZSTD_LEVEL} -q -c "
        f"| split -b {part_size} -d -a 3 - {sh(out / (ARCHIVE + '.part-'))}"
    )
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    lst.unlink()
    if r.returncode != 0:
        die(f"tar|zstd|split rc {r.returncode}: {r.stderr.strip()[-600:]}")
    parts = sorted(out.glob(ARCHIVE + ".part-*"))
    if not parts:
        die(input_missing("no archive parts"))
    return parts


def read_archive(parts: list[Path]) -> dict[str, tuple[int, str]]:
    """{member name: (size, sha256)} read back from the parts as one stream (cat | zstd -d | tar) — the bytes the
    receiver gets. A non-regular member dies."""
    p = subprocess.Popen(
        ["bash", "-c", "set -o pipefail; cat " + " ".join(sh(x) for x in parts) + " | zstd -dc -q"],
        stdout=subprocess.PIPE,
    )
    out = {}
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                die(f"non-regular member {m.name} in the archive (type {m.type!r})")
            h, f = hashlib.sha256(), tf.extractfile(m)
            while chunk := f.read(CHUNK):
                h.update(chunk)
            out[m.name] = (m.size, h.hexdigest())
    if p.wait() != 0:
        die(f"reading the parts back failed (rc {p.returncode})")
    return out
