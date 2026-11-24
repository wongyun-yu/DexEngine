"""Shared constants and helpers of the SPIDER input steps (standard library only — imported by SPIDER's python too)."""

from __future__ import annotations

import ast
import csv
import hashlib
import os
import re
from pathlib import Path

#: Code tree whose owner constants are read (by AST — this file also runs under SPIDER's isolated python).
OV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


DATA = ""  # data root (--data), set by the command line


ROBOT = "wuji2"


OWNER_MASS = f"{OV}/dexengine/rl/tasks/cfg/residual.py"


SCENES = ("scene.xml", "scene_eq.xml")


def die(msg):
    raise SystemExit(f"[E5_SPIDER_IN] FAIL {msg}")


def owner_const(path, name):
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return ast.literal_eval(n.value)
    die(f"{name} not found in {path}")
    return None


def read_clips(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or any(k not in rows[0] for k in ("clip", "dataset", "embodiment")):
        die(f"{path}: needs columns clip, dataset, embodiment")
    if len({r["clip"] for r in rows}) != len(rows):
        die(f"{path}: duplicate clips")
    return rows


def tree_of(r):
    return f"{DATA}/{r['dataset']}/base/{ROBOT}/{r['embodiment']}/{r['clip']}"


def out_of(out, r):
    return f"{out}/processed/{r['dataset']}/{ROBOT}/{r['embodiment']}/{r['clip']}"


def task_objects(ti):
    objs = ti.get("objects")
    objs = ast.literal_eval(objs) if isinstance(objs, str) else objs
    if not objs:
        die(f"{ti.get('clip_id')}: task_info has no objects")
    return objs


def articulated(ti, name):
    a = ti.get("articulated_object")
    a = ast.literal_eval(a) if isinstance(a, str) else a
    return bool(a) and f"{a.get('side')}_object" == name


def meshdir_root(clip_dir, xml_text):
    md = re.findall(r'<compiler[^>]*meshdir="([^"]+)"', xml_text)
    if len(md) != 1:
        die(f"{clip_dir}: {len(md)} meshdir attributes")
    return os.path.normpath(os.path.join(clip_dir, md[0]))


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()
