"""Input fingerprint of the retargeting solve (generate-xml · ik) — the single owner.

Why. If skipping (batch --skip-existing) looked only at whether output files exist, it would reuse outputs solved from
old inputs after the inputs (MANO layer · object layer · hand rig) changed. So every finished solve writes the
fingerprint of the inputs that clip read next to its outputs, and skipping reuses an output only when the fingerprint
measured from the current inputs equals the recorded one. No fingerprint (outputs older than this record) means
unknown = solve again (no silent reuse).

Fingerprint (all content hashes — no paths or mtimes; the stage's MANO is a symlink and is followed)
  code     solve code version (SOLVE_VERSION — bumped by hand for changes that alter scene · ik results)
  mano     every file under the clip's MANO folder (processed/<ds>/mano/<emb>/<task>) (the folder, not a list — files
           read later cannot be missed)
  objects  the object assets the clip's MANO task_info points at (every string value that is an existing path
           relative to the data tree · `<count>:<hash>` · the hand rig place is excluded — see rig)
  rig      the hand rig the solve uses = the rig folder of the bundle (assets/robots/hands/<hand>/mjcf — the batch
           mirrors it into the data tree on every run · posture prior file included)
  scene    (ik only) the scene XML generate-xml wrote — a rebuilt scene makes ik stale too
Record: one RECORD file in the robot clip folder (<robot>/<emb>/<task>), one entry per phase; written with
tmp + replace (a file shared by hard link with an adopted layer is never edited in place).
Verdict: fresh (same) · stale (different) · unknown (no record). Dependencies: standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

#: Solve code version — bumped by hand for changes that alter the results of dexengine.retarget.scene · ik.
SOLVE_VERSION = "2026-10-12a"
#: Record file name (inside the robot clip folder) · the phases that are fingerprinted.
RECORD = "retarget_input.json"
PHASES = ("generate-xml", "ik")
#: Names skipped when measuring an asset folder — plugin state and bytecode are not assets (the batch's rig copy skips
#: the same names).
IGNORE = (".omc", "__pycache__")

_CACHE: dict[str, str] = {}
_LOCK = threading.Lock()


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(root: Path, pattern: str = "") -> str:
    """Content fingerprint of one folder (or file) — per file (root-relative path, content sha256) · symlinks are
    followed · IGNORE names are skipped. With `pattern`, only files of that glob directly below the root. A missing root
    is "absent" (never equal to an existing one)."""
    root = Path(root)
    if root.is_file():
        return _sha_file(root)
    if not root.is_dir():
        return "absent"
    if pattern:
        files = sorted(q for q in root.glob(pattern) if q.is_file())
    else:
        files = []
        for dp, dn, fn in os.walk(root, followlinks=True):
            dn[:] = sorted(d for d in dn if d not in IGNORE)
            files += [Path(dp) / f for f in fn if f not in IGNORE]
        files.sort()
    h = hashlib.sha256()
    for q in files:
        h.update(q.relative_to(root).as_posix().encode() + b"\0" + bytes.fromhex(_sha_file(q)))
    return h.hexdigest()[:32]


def _cached(root: Path) -> str:
    """Folders shared by many clips (objects · rig) are measured once per process."""
    key = os.path.realpath(root)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    v = tree_sha(root)
    with _LOCK:
        return _CACHE.setdefault(key, v)


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def object_refs(task_info: dict, dataset_dir: Path, dataset: str) -> list[str]:
    """The assets the MANO task_info points at — every string value that is an existing path relative to the data tree
    (`dataset_dir`) (a rule, not a key list — new keys or asset places are not missed). Single-part values (names) and
    values pointing outside are not paths. The hand rig place (assets/robots) is measured by the rig entry."""
    rig = f"processed/{dataset}/assets/robots/"
    out = set()
    for s in _strings(task_info):
        n = os.path.normpath(s) if s else ""
        parts = n.split("/")
        if len(parts) < 2 or os.path.isabs(n) or ".." in parts or (n + "/").startswith(rig):
            continue
        if (Path(dataset_dir) / n).exists():
            out.add(n)
    return sorted(out)


def current(dataset_dir: Path, dataset: str, robot: str, emb: str, task: str, phase: str, rig_src: Path) -> dict:
    """The fingerprint measured from the current inputs (phase `phase`). An unreadable MANO task_info raises — there is
    no solve without inputs."""
    if phase not in PHASES:
        raise ValueError(f"dexengine.retarget.input_fp: unknown phase {phase!r} (known {PHASES})")
    dataset_dir = Path(dataset_dir)
    mano = dataset_dir / "processed" / dataset / "mano" / emb / task
    info = json.loads((mano / "task_info.json").read_text(encoding="utf-8"))
    h, refs = hashlib.sha256(), object_refs(info, dataset_dir, dataset)
    for rel in refs:
        h.update(rel.encode() + b"\0" + _cached(dataset_dir / rel).encode() + b"\n")
    fp = {
        "code": SOLVE_VERSION,
        "mano": tree_sha(mano),
        "objects": f"{len(refs)}:{h.hexdigest()[:32]}",
        "rig": _cached(Path(rig_src)),
    }
    if phase == "ik":
        fp["scene"] = tree_sha(dataset_dir / "processed" / dataset / robot / emb / task, "*.xml")
    return fp


def recorded(task_dir: Path, phase: str) -> dict | None:
    """The recorded fingerprint (that phase's entry) — None (unknown) when the file is missing, unreadable or lacks the
    entry."""
    try:
        v = json.loads((Path(task_dir) / RECORD).read_text(encoding="utf-8")).get(phase)
    except (OSError, ValueError, AttributeError):
        return None
    return v if isinstance(v, dict) else None


def verdict(task_dir: Path, phase: str, cur: dict) -> str:
    rec = recorded(task_dir, phase)
    return "unknown" if rec is None else "fresh" if rec == cur else "stale"


def _write(task_dir: Path, rec: dict) -> None:
    f = Path(task_dir) / RECORD
    if not rec:
        f.unlink(missing_ok=True)
        return
    tmp = f.with_name(f"{RECORD}.tmp{os.getpid()}_{threading.get_ident()}")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, f)


def _all(task_dir: Path) -> dict:
    try:
        rec = json.loads((Path(task_dir) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return rec if isinstance(rec, dict) else {}


def stamp(task_dir: Path, phase: str, cur: dict) -> None:
    """After a solve, write the fingerprint into that phase's entry (other entries stay)."""
    _write(task_dir, {**_all(task_dir), phase: cur})


def forget(task_dir: Path, phase: str) -> None:
    """Before solving again, clear that phase's entry and the later ones — if the solve dies midway, an old
    fingerprint never vouches for new outputs."""
    rec = _all(task_dir)
    for p in PHASES[PHASES.index(phase) :]:
        rec.pop(p, None)
    _write(task_dir, rec)


def recorded_counts(robot_dir: Path) -> tuple[int, int]:
    """(robot clip folders, of which carry a record file) — counts <robot>/<emb>/<task>. Used for one stage-level
    question only: "can every clip here be judged by the per-clip rule (verdict)?". When equal (> 0) stage 1 does not
    clear an unstamped stage — reuse is then decided per clip by the solve's --skip-existing from the current inputs
    (nothing is judged correct here)."""
    tasks = [t for e in sorted(Path(robot_dir).iterdir()) if e.is_dir() for t in sorted(e.iterdir()) if t.is_dir()]
    return len(tasks), sum(1 for t in tasks if _all(t))


if __name__ == "__main__":  # --recorded <robot folder> → "<clips> <with record>" · rc 0 = equal and > 0
    import sys

    if len(sys.argv) != 3 or sys.argv[1] != "--recorded" or not Path(sys.argv[2]).is_dir():
        sys.exit("dexengine.retarget.input_fp: usage --recorded <the stage's robot folder>")
    n, have = recorded_counts(Path(sys.argv[2]))
    print(n, have)
    sys.exit(0 if n and n == have else 1)
