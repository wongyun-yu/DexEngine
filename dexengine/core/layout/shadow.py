"""Shadow data roots: an overlay whose writable folders are real and whose other entries link into a base root;
a process on a shadow root may not write into the base's data layers."""

from __future__ import annotations

import json
import os

from dexengine.core.errors import InputMissingError
from dexengine.core.layout.generations import GENERATION_FILE
from dexengine.core.layout.layers import _within, data_tops

ROOT_TAG = "[DATA_ROOT]"


# ── Augmentation shadow root: an overlay on the canonical data root (the base) whose writable folders are real ──
# Why: the batch-augmentation rollout (dexengine.record.augment.scene.init_shadow · shadow_dir) reads variant clips
# that live only in its overlay and everything else through links into the canonical root — an overlay
# check_data_root rejects. A registered shadow states its base in SHADOW_FILE (writer scene.init_shadow · reader
# data_shadow): its links may reach only that base, and a process whose data root is a shadow dies on a write that
# lands in the base's data layers (shadow_guard — an audit hook: reading through the links is the point, writing
# through one is the leak). A shadow inside a collect work piece (<out>/aug/shadow · <out> = run layer
# run_collect_work) writes its piece freely — run layers are not data layers (data_tops), so no exemption is needed.
SHADOW_FILE = "DATA_SHADOW.json"
SHADOW_PURPOSES = ("aug_place", "e5_kin")  # e5_kin = E5 kinematic-baseline tree root (built only by kin_data/prep.py)
_SHADOW_GUARD = "_dexengine_data_shadow_guard"  # sys.modules key — one hook per process (this module loads twice)
_WRITE_EVENTS = {
    "os.rename": (0, 1),
    "os.remove": (0,),
    "os.rmdir": (0,),
    "os.mkdir": (0,),
    "os.symlink": (1,),
    "os.link": (1,),
    "os.truncate": (0,),
    "shutil.copyfile": (1,),
    "shutil.copytree": (1,),
    "shutil.rmtree": (0,),
}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def shadow_record(base: str, purpose: str) -> dict:
    """The shadow identity record (the content of SHADOW_FILE). Dies on an unknown purpose, or when the base is not a
    directory or is itself a generation or a shadow."""
    b = os.path.realpath(base)
    if purpose not in SHADOW_PURPOSES:
        raise SystemExit(f"{ROOT_TAG} ✗ shadow purpose {purpose!r} — registered: {SHADOW_PURPOSES}")
    if not os.path.isdir(b) or any(os.path.isfile(os.path.join(b, f)) for f in (GENERATION_FILE, SHADOW_FILE)):
        raise InputMissingError(
            f"{ROOT_TAG} ✗ shadow base {b} is not a canonical data root (missing · generation · shadow)"
        )
    return {"kind": "shadow", "purpose": purpose, "base": b}


def write_shadow_record(root: str, base: str, purpose: str) -> dict:
    """Write the identity of shadow root `root` (writer scene.init_shadow — rewritten when a move to another machine
    changes the base)."""
    rec = shadow_record(base, purpose)
    # a tmp unique to each writer — with a fixed name two concurrent writers would move each other's tmp
    tmp = os.path.join(str(root), f"{SHADOW_FILE}.tmp.{os.getpid()}.{os.urandom(4).hex()}")
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, os.path.join(str(root), SHADOW_FILE))
    return rec


def data_shadow(root: str) -> dict | None:
    """The shadow identity of root `root` — None without SHADOW_FILE; otherwise it must equal shadow_record."""
    f = os.path.join(str(root), SHADOW_FILE)
    if not os.path.isfile(f):
        return None
    if os.path.isfile(os.path.join(str(root), GENERATION_FILE)):
        raise SystemExit(f"{ROOT_TAG} ✗ {root}: holds both a generation record and a shadow record")
    with open(f) as fh:
        rec = json.load(fh)
    want = shadow_record(rec.get("base", ""), rec.get("purpose", "")) if isinstance(rec, dict) else None
    if rec != want:
        raise SystemExit(f"{ROOT_TAG} ✗ {f}: record {rec} ≠ {want}")
    return rec


def shadow_guard(root: str, base: str) -> None:
    """For a process whose data root is a shadow (children re-arm it through env): a write that lands in the base's
    data layers (data_tops) → one stderr line + exit 3. A collect work piece (run layer run_collect_work, which holds
    the shadow) is outside the data layers and needs no exemption. mkdir (exist_ok) of an existing directory is not a
    write. Marker (stderr, once): `[DATA_ROOT] shadow <purpose> <root> · base <base> · base write = death`."""
    import sys
    import types

    if _SHADOW_GUARD in sys.modules:
        return
    sys.modules[_SHADOW_GUARD] = types.ModuleType(_SHADOW_GUARD)
    tops = tuple(os.path.join(base, t) for t in sorted(data_tops()))

    def hit(ev: str, p) -> str:
        if p is None or isinstance(p, int):
            return ""
        rp = os.path.realpath(os.fsdecode(p))
        if ev == "os.mkdir" and os.path.isdir(rp):
            return ""
        return rp if any(_within(rp, t) for t in tops) else ""

    def hook(ev: str, args: tuple) -> None:
        if ev == "open":
            mode, flags = args[1], args[2]
            w = (
                any(c in mode for c in "wax+")
                if isinstance(mode, str)
                else bool(isinstance(flags, int) and flags & _WRITE_FLAGS)
            )
            ps = (args[0],) if w else ()
        else:
            ps = tuple(args[i] for i in _WRITE_EVENTS.get(ev, ()))
        for p in ps:
            if rp := hit(ev, p):
                sys.stderr.write(
                    f"{ROOT_TAG} ✗ shadow root {root} writes into the data of base {base} ({ev} {rp}) — "
                    "writable folders are the real directories inside the shadow (scene.shadow_dir)\n"
                )
                sys.stderr.flush()
                os._exit(3)

    sys.addaudithook(hook)
    sys.stderr.write(f"{ROOT_TAG} shadow {data_shadow(root)['purpose']} {root} · base {base} · base write = death\n")
