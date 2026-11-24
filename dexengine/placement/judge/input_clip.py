"""Clip-level fields of the judge input record: setup · case · contact · base · objcol, measured on the clip's base
tree (the same functions the judge and the bake call)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core import stdio
from dexengine.core.errors import input_missing
from dexengine.placement.judge.input_fields import (
    _CANON,
    CLIP_FIELDS,
    D40_SEP,
    ERR,
    SETUP_ROUND,
    SUP_SEP,
    _ar,
    _coords,
    _h,
    _idle_arm,
    _objhull,
    _ql,
    _sf,
)


def base_tree(ds: str, emb: str, hand: str, clip: str) -> Path:
    """The base tree the judge reads the case from — paths.ref_base_dir (the judge's _CASE_ROOT function)."""
    return P.ref_base_dir(ds, hand, emb) / clip


def _setup(tree: Path, ti: dict, scene: str) -> str:
    C = _coords()
    with C.scene_env(scene), stdio.capture():  # scene env owner: coords.scene_env
        z = float(C.scene_z_offset(tree))
        sup = C.compute_aerial_support(tree) if scene == "aerial" else C.compute_desk_support(tree, ti)
    r = SETUP_ROUND
    ds_ = sorted(
        ",".join([f"{float(d[k]):.{r}f}" for k in ("x", "y", "z", "r")] + [str(d["kind"])]) for d in sup["discs"]
    )
    ex = sorted(f"{x['obj']}/{x['why']}" for x in sup["excluded"])
    sp = sorted(
        f"{x['obj']}>{x['main']}>{int(bool(x['holds']))}@" + ",".join(f"{float(v):.{r}f}" for v in x["pose"])
        for x in sup["supports"]
    )
    # supports and exclusions go after `~` separators — `|` separates record fields and cannot appear inside one
    return (
        f"{scene}:{z:.{r}f}:"
        + ";".join(ds_)
        + (SUP_SEP + ";".join(sp) if sp else "")
        + (D40_SEP + ",".join(ex) if ex else "")
    )


def _objcol_digests(ds: str, keys: list) -> dict:
    """{(object id, articulated): collision_parts digest} — cache key = the USD file (mtime, size) (re-measured after a
    re-bake). With pxr and torch the function runs here, otherwise through its CLI (`hull.py --digest`) in the
    retarget python (paths.retarget_py). Failure raises (clip_parts writes the error into the field)."""

    def ck(k):
        f = _objhull().usd_file(ds, *k)
        st = f.stat() if f.is_file() else None
        return ("objcol", ds) + k + ((st.st_mtime_ns, st.st_size) if st else (None, None))

    want = [k for k in dict.fromkeys(keys) if ck(k) not in _CANON]
    if want and importlib.util.find_spec("pxr") and importlib.util.find_spec("torch"):
        for k in want:
            _CANON[ck(k)] = _objhull().collision_parts(ds, *k).digest
    elif want:
        import subprocess  # noqa: PLC0415

        args = [f"{oid}:{int(art)}" for oid, art in want]
        hull_py = Path(_objhull().__file__)
        r = subprocess.run(
            [P.retarget_py(), str(hull_py), "--digest", ds, *args], capture_output=True, text=True, timeout=600
        )
        if r.returncode != 0:
            raise RuntimeError(f"dexengine.data.objects.hull --digest rc {r.returncode}: {r.stderr.strip()[-300:]}")
        got = json.loads(r.stdout.strip().splitlines()[-1])
        for k, a in zip(want, args, strict=True):
            _CANON[ck(k)] = got[a]
    return {k: _CANON[ck(k)] for k in keys}


def _objcol_rows(ti: dict) -> list:
    """Objects the objcol field covers — the main and support objects the simulator spawns (name · asset id · art.)."""
    return sorted(
        (str(o["name"]), _objhull().asset_id(o), bool(o.get("articulation")))
        for o in ti.get("objects") or []
        if o.get("role") in ("main", "support")
    )


def _objcol(ti: dict, ds: str) -> str:
    rows = _objcol_rows(ti)
    if not rows:
        return "none"
    dg = _objcol_digests(ds, [(oid, art) for _n, oid, art in rows])
    return _h([(n, oid, art, dg[(oid, art)]) for n, oid, art in rows])


def _case(tree: Path, ti: dict, active_mode: str) -> str:
    lay = _ql().read(str(tree), use_cache=False)
    hands = tuple(s for s in ("right", "left") if s in lay.hands)
    if active_mode == "both":  # the judge ran in two-hand mode — record the sides it actually saw
        act, idle = ("right", "left"), ()
    else:
        act, idle, _rule = _idle_arm().active_idle_for_tree(ti.get("case"), hands, tuple(lay.manip_sides()))
    per = ((ti.get("rules") or {}).get(_ar().ASSIGN_KEY) or {}).get("per_side") or {}
    roles = sorted(
        (str(o.get("name")), str(o.get("role")), str(o.get("side")), str(o.get("obj_id")), bool(o.get("articulation")))
        for o in ti.get("objects") or []
    )
    return _h(
        {
            "case": (ti.get("case") or {}).get("case"),
            "active": list(act),
            "idle": list(idle),
            "assign": {s: [bool(p.get("active")), p.get("main"), p.get("tag")] for s, p in sorted(per.items())},
            "roles": roles,
        }
    )


def _contact(tree: Path, ti: dict, ds: str) -> str:
    import numpy as np  # noqa: PLC0415 — lazy: the training entry imports this module before Kit starts

    SF = _sf()
    with np.load(tree / "0/trajectory_kinematic.npz") as z:
        kin = z["qpos"]
    spec = P.hand_spec_dir(P.hand_id(str(ti.get("hand"))))
    tin = SF.TreeInputs(P.main_root(), ds, tree.name, ti, kin, spec)
    rows = []
    for s in SF.sides_for(ti, SF.LAYER_BASE):
        present = P.sidecar_file("contact_base", ds, tree.name, s, spec).is_file()
        rows.append((s, SF.input_id(tin.expected(SF.LAYER_BASE, s)), present))
    return _h(rows) if rows else "none"


def _base(tree: Path) -> str:
    import numpy as np  # noqa: PLC0415 — lazy: the training entry imports this module before Kit starts

    with np.load(tree / "0/trajectory_kinematic.npz") as z:
        q = np.ascontiguousarray(z["qpos"])
    h = hashlib.sha256()
    for part in (str(q.dtype).encode(), str(q.shape).encode(), q.tobytes()):
        h.update(part)
    return h.hexdigest()[:12]


def base_part(ds: str, emb: str, hand: str, clip: str) -> str:
    """The `base` field alone (qpos hash of the clip's base tree) — for callers that compare only the base
    (dexengine.placement.train_ready). No base tree → "" (no current base to compare), not measurable →
    `error:<exception type>`."""
    tree = base_tree(ds, emb, hand, clip)
    if not (tree / "0/trajectory_kinematic.npz").is_file():
        return ""
    try:
        return _base(tree)
    except Exception as e:  # noqa: BLE001 — written into the field (counted as unevaluable)
        return f"{ERR}{type(e).__name__}"


def clip_parts(
    ds: str, emb: str, hand: str, clip: str, scene: str, active_mode: str = "auto", strict: bool = False
) -> dict:
    """Clip fields {setup, case, contact, base, objcol} from one base tree.
    strict (judge): a field that cannot be measured prints a `[JUDGE_FP] ✗` reason and re-raises its exception.
    Otherwise (the comparison's current value) the field gets `error:<exception type>` (diff counts it unevaluable)."""
    out: dict = {}
    tree = base_tree(ds, emb, hand, clip)

    def bad(k: str, e: BaseException) -> str:
        if strict:
            print(
                input_missing(
                    f"[JUDGE_FP] ✗ {clip}: judge input field {k} cannot be measured on this machine — "
                    f"{type(e).__name__}: {str(e).strip()[-400:]} — dies without judging"
                ),
                flush=True,
            )
            raise e
        return f"{ERR}{type(e).__name__}"

    try:
        ti = json.loads((tree / "task_info.json").read_text())
    except (OSError, ValueError) as e:
        bad("base", e)  # strict dies here
        return {k: f"{ERR}no_base:{type(e).__name__}" for k in CLIP_FIELDS}
    for k, fn in (
        ("setup", lambda: _setup(tree, ti, scene)),
        ("case", lambda: _case(tree, ti, active_mode)),
        ("contact", lambda: _contact(tree, ti, ds)),
        ("base", lambda: _base(tree)),
        ("objcol", lambda: _objcol(ti, ds)),
    ):
        try:
            out[k] = fn()
        except (Exception, SystemExit) as e:  # noqa: BLE001, PERF203 — strict dies · otherwise written into the field
            out[k] = bad(k, e)
    return out


def inputs(ds: str, emb: str, hand: str, clip: str) -> list:
    """The **data** files the clip fields open — the cache key ((mtime, size)): coords.setup_inputs (inputs of the
    setup and sidecar fingerprints) plus the sidecar files themselves (the contact field's presence) and the objcol
    USD files."""
    tree = base_tree(ds, emb, hand, clip)
    ti_p = tree / "task_info.json"
    try:
        ti = json.loads(ti_p.read_text())
    except (OSError, ValueError):
        return [ti_p]
    out = list(_coords().setup_inputs(tree, ti))
    spec = (
        P.hand_spec_dir(P.hand_id(str(ti["hand"]))) if ti.get("hand") else ""
    )  # unknown = '' (a registered clip dies)
    out += [P.sidecar_file("contact_base", ds, clip, s, spec) for s in ("right", "left")]
    out += [_objhull().usd_file(ds, oid, art) for _n, oid, art in _objcol_rows(ti)]  # the objcol USD files
    return out
