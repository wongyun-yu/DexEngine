#!/usr/bin/env python3
"""A trained tree's **content identity** — the one owner of "did this clip's training data change?" (the RL owner
ledger's carry split reads it). It chooses **whom to judge**; the judge stays the
carry criterion (STATUS — a changed clip carries only when the old checkpoint judged on the new tree is in the band).

Why not clip_gen.tree_sha: that is sha256 of the task_info.json bytes, and task_info carries build stamps — wall-clock
times (provenance.time · reference.time), version strings (rules.*.version · provenance.adapter_version …) and input
fingerprints (rules.WRIST_SLACK.fp hashes a sidecar file that itself carries build times). Rebuilding a clip from the
same inputs changes tree_sha every time, so 'same tree_sha' cannot say 'same data'.

What a tree's content is — **what the training env and the RL reference read** (structure, not a list)
  task_info   the JSON with every **stamp** and every **record** removed, recursively:
                version  a dict key `version` or `*_version`
                time     a string value that is an ISO-8601 date-time (any key — a new time field is caught too)
                source   the top-level `provenance` subtree (who built it, from what, when) · a dict key `fp` or `*_fp`
                         (a stage's input fingerprint — the inputs' effect is in the arrays compared below) · a string
                         value that is an absolute path (the machine location it was built at)
                record   a subtree named in RECORD_PATHS — what a rule run wrote down about the clip (its gate's
                         verdict · its label · a dataset statistic · a quality measurement). The one statement of
                         'record' is that tuple; nothing the env, the slot rule or the reference loader reads is in it
                empty    a dict value that is (or became) `{}` or `[]` — absent and empty read the same (`get() or []`)
              Every other field is meaning: a field never seen before is compared, so a difference there = changed
              (no silent pass). NaN is compared as itself (json 'NaN').
  scene       scene_parts.SCENE_XMLS bytes (the MJCF the env loads) — absent/present is compared too
  arrays      tree_arrays(): `0/<name>.npz` whose name has no further dot (a dotted name is a repair backup or an
              unfinished write) — the file list, every key, dtype, shape and raw bytes, **except stamp keys**: the
              contact sidecar's input fingerprint (dexengine.data.contact.fingerprint.FP_KEYS — read from its owner; it
              carries the build time) and a key that is a stamp by the same form as above (version key · a scalar string
              that is a date-time or an absolute path) **and the rollout file's arrays**: the reference loader finds a
              clip by the presence of ROLLOUT_NPZ and reads its arrays from REFERENCE_NPZ, so where the tree has the
              reference file the rollout counts as present/absent only; a tree without the reference file is read from
              the rollout (the loader's fallback) and there its bytes count · **and the reference file's UNREAD_KEYS**
              (`qpos_rollout` — the rollout trajectory the rollout file carries: no env · reference loader · judge code
              reads it — the selftest reads source/ to prove it): present/absent only, like the rollout file
  cmp         the comparator's version CMP, written into parts() unhashed — a stored snapshot row says which rule
              measured it. A row without it (LEGACY — measured before UNREAD_KEYS) is compared with the tree now
              measured by that same rule (parts(tree, version_of(row))), so an open legacy snapshot splits exactly as
              before (its old trees are mostly overwritten — nothing to re-measure). An unknown value dies.
              LEGACY expires when no legacy snapshot row is open (then this branch goes)
  window      the clip's time window on its parent (parent_clip, segment.start, segment.end) — kept apart so the split
              can tell 'segment moved' (retrain) from 'content changed' (judge)
Not content: other files in the tree (0/*.json reports · repair backups) — the env does not read them.
Same = exact: every compared byte equal (`_array_id` is the one place an array's identity is stated — no tolerance).

  parts(tree)          {part: sha256} — 'task_info' · 'scene:<file>' · 'npz:<file>' (+ 'window' JSON, not hashed)
  diff(old, new)       the first differing part in sorted order, '' = same content
  view(ti)             the stamp- and record-free task_info (what 'task_info' hashes) — the selftest plants on it
  ignored_text()       what was left out since import — printed once at process exit as the run's marker
                       (`[TREE_CONTENT] ignore fp=… stamp=… record=… unread=…`; a measuring run without it is void)
Stdlib + numpy + scene_parts + dexengine.data.contact.fingerprint's constant (no hand env — the queue CLI reads it
without one).
"""

from __future__ import annotations

import atexit
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from dexengine.data.contact import fingerprint as SF  # FP_KEYS — the contact sidecar's fingerprint keys (owner)
from dexengine.data.objects import scene_parts as SP  # SCENE_XMLS — the scene files a tree carries (owner)

TAG = "[TREE_CONTENT]"
ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?")
SOURCE_TOP = "provenance"
#: task_info subtrees that are a rule run's **record** — the one statement (slash path from the root; a trailing `*`
#: = every key with that prefix). Each is written by a rule about the clip and read by no env · slot rule · reference
#: loader: the assignment gate's verdict · the contact-role rule's label and before/after · the adapter's own main-hand
#: note · the dataset's hand-direction statistics · the retarget quality measurements · the case stage's move layer
#: (case.move_runs — read only by the D36 cut rule; its effect on a clip is the cut windows, i.e. the pieces' own
#: arrays — the moved-object switch rule). A path not named here is content.
RECORD_PATHS = (
    "rules/ASSIGN/gate",
    "rules/D23",
    "case/main_by_adapter",
    "case/move_runs",
    "frame/hand_dir_*",
    "quality",
)
#: The reference loader's two trajectory files (rl/tasks/mdp/trajectory/loader.py; the statement of what a run reads —
#: dexengine/placement/bundle/run_inputs.py — names the rollout `kit:loader-discovery`): module doc 'arrays'.
REFERENCE_NPZ, ROLLOUT_NPZ = "trajectory_kinematic.npz", "trajectory_ikrollout.npz"
PRESENT = "present"
#: Reference-file keys nobody that trains or judges reads (module doc 'arrays') — counted by presence only.
UNREAD_KEYS = {REFERENCE_NPZ: ("qpos_rollout",)}
#: The comparator's version (module doc 'cmp') · LEGACY = a stored row without it (measured before UNREAD_KEYS).
CMP_KEY, CMP, LEGACY = "cmp", "2", ""
#: What parts() left out since import — fp: fingerprint keys · stamp: other stamp keys (arrays) · record: subtrees ·
#: unread: rollout files counted by presence only.
IGNORED: collections.Counter = collections.Counter()


def is_version_key(k: str) -> bool:
    return k == "version" or k.endswith("_version")


def is_fp_key(k: str) -> bool:
    return k == "fp" or k.endswith("_fp")


def is_stamp_value(v) -> bool:
    """A string that is a wall-clock date-time or an absolute path (build time · build location)."""
    return isinstance(v, str) and (bool(ISO_TIME.fullmatch(v)) or v.startswith("/"))


def is_record_path(p: str) -> bool:
    return any(p.startswith(r[:-1]) if r.endswith("*") else p == r for r in RECORD_PATHS)


def _strip(v, at: str = ""):
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            p = f"{at}/{k}" if at else k
            if is_record_path(p):
                IGNORED["record"] += 1
            elif not (is_version_key(k) or is_fp_key(k) or is_stamp_value(x)):
                y = _strip(x, p)
                if not (isinstance(y, (dict, list)) and not y):
                    out[k] = y
        return out
    if isinstance(v, list):
        return [_strip(x, f"{at}/#") for x in v if not is_stamp_value(x)]
    return v


def view(ti: dict) -> dict:
    """task_info without its stamps and records (module doc) — the top-level provenance subtree goes whole."""
    return _strip({k: v for k, v in ti.items() if k != SOURCE_TOP})


def ignored_text() -> str:
    return (
        f"ignore fp={IGNORED['fp']} stamp={IGNORED['stamp']} record={IGNORED['record']} "
        f"unread={IGNORED['unread']} unread_key={IGNORED['unread_key']} (FP_KEYS {len(SF.FP_KEYS)} · "
        f"RECORD_PATHS {len(RECORD_PATHS)}) · version cmp={CMP} {IGNORED['cmp']} · legacy (no marker) "
        f"{IGNORED['legacy']}"
    )


def version_of(stored: dict) -> str:
    """The comparator version a stored parts dict was measured by — no key = LEGACY · an unknown value dies."""
    v = stored.get(CMP_KEY, LEGACY)
    if v not in (CMP, LEGACY):
        raise ValueError(
            f"content fingerprint comparator version {v!r} — only {CMP!r} and no marker (legacy) are known"
        )
    return v


@atexit.register
def _say_ignored() -> None:
    """The marker of a run that measured content — printed once at exit, so no caller can leave it out."""
    if IGNORED:
        print(f"{TAG} {ignored_text()}", flush=True)


def _canon(v) -> bytes:
    return json.dumps(v, sort_keys=True, ensure_ascii=False).encode()


def window(ti: dict) -> str:
    seg = ti.get("segment") or {}
    return json.dumps([ti.get("parent_clip") or ti.get("clip_id"), seg.get("start"), seg.get("end")])


def tree_arrays(tree: Path) -> list[Path]:
    """A tree's arrays — `0/<name>.npz` whose name has no further dot (`<file>.<tag>.npz` = a repair's backup ·
    `<file>.tmp.npz` = an unfinished write). The one statement — segment.build.tree_arrays reads it."""
    return sorted(f for f in (Path(tree) / "0").glob("*.npz") if "." not in f.name[: -len(".npz")])


def _array_id(a: np.ndarray) -> bytes:
    """An array's identity — its exact bytes (an object array: its canonical JSON)."""
    return _canon(a.tolist()) if a.dtype.hasobject else np.ascontiguousarray(a).tobytes()


def _stamp_key(k: str, a: np.ndarray) -> str:
    """'fp' | 'stamp' | '' — why an array key is not content (module doc 'arrays')."""
    if k in SF.FP_KEYS:
        return "fp"
    scalar_text = a.ndim == 0 and (a.dtype.kind in "US" or a.dtype.hasobject)
    return "stamp" if is_version_key(k) or (scalar_text and is_stamp_value(a.item())) else ""


def _npz_sha(f: Path, unread: tuple[str, ...] = ()) -> str:
    h = hashlib.sha256()
    with np.load(f, allow_pickle=True) as z:
        for k in sorted(z.files):
            if k in unread:  # only presence counts (module doc 'arrays', UNREAD_KEYS)
                IGNORED["unread_key"] += 1
                h.update(k.encode() + b"\0" + PRESENT.encode() + b"\0")
                continue
            a = z[k]
            why = _stamp_key(k, a)
            if why:
                IGNORED[why] += 1
                continue
            h.update(k.encode() + b"\0" + a.dtype.str.encode() + b"\0" + repr(a.shape).encode() + b"\0")
            h.update(_array_id(a))
    return h.hexdigest()


def parts(tree: Path, cmp: str = CMP) -> dict[str, str]:
    """{part: sha256} of a tree's content + 'window' (+ CMP_KEY when cmp is CMP) by the comparator version cmp
    (module doc 'cmp' — LEGACY only to compare with a stored legacy row). A tree without task_info.json dies."""
    if cmp not in (CMP, LEGACY):
        raise ValueError(f"comparator version {cmp!r} — {CMP!r} or legacy only")
    tree = Path(tree)
    f = tree / "task_info.json"
    if not f.is_file():
        raise ValueError(f"{f}: no task_info.json — the content cannot be measured")
    ti = json.loads(f.read_text())
    out = {"task_info": hashlib.sha256(_canon(view(ti))).hexdigest(), "window": window(ti)}
    for x in SP.SCENE_XMLS:
        p = tree / x
        out[f"scene:{x}"] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "-"
    arrs = tree_arrays(tree)
    if not arrs:
        raise ValueError(f"{tree}: no tree arrays (0/*.npz) — the content cannot be measured")
    has_reference = any(a.name == REFERENCE_NPZ for a in arrs)
    for a in arrs:
        if a.name == ROLLOUT_NPZ and has_reference:
            IGNORED["unread"] += 1
            out[f"npz:{a.name}"] = PRESENT
        else:
            out[f"npz:{a.name}"] = _npz_sha(a, UNREAD_KEYS.get(a.name, ()) if cmp == CMP else ())
    IGNORED["cmp" if cmp == CMP else "legacy"] += 1
    if cmp == CMP:
        out[CMP_KEY] = CMP
    return out


def diff(old: dict[str, str], new: dict[str, str]) -> str:
    """First differing part (sorted · a part on one side only counts) — '' = the same content."""
    for k in sorted(set(old) | set(new)):
        if old.get(k) != new.get(k):
            return k
    return ""
