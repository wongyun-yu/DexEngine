"""Concrete runs of a stage-1 step, and the rig/prior identity of a dataset's outputs.

``emit_run`` turns one step of ``table.STAGES`` into argv lists (per embodiment / side as the step's axis says,
only for embodiments whose base layer exists). ``emit_rig`` reports the rig identity three ways -- declared
(registry), measured (``task_info.hand_asset_sha256_xml3`` the base adapter writes) and stamped (the stage-1 stamps) --
and ``invalidate_stage_stamps`` removes stage stamps that no longer match the current rig and prior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data import rules as RR
from dexengine.data.stage1 import table as T

#: Rig and prior stamp file names, kept next to each other in the base-layer root and the retarget stage folder.
RIG_STAMP_NAME = ".stage1_rig_sha256"
PRIOR_STAMP_NAME = ".stage1_prior_sha256"


def f0_axis(ds: str) -> dict:
    """The frame-0 axis of step 7 = the dataset's judge axis (its profile variant); one function for the writer
    (emit_run) and the counter (f0_cells)."""
    return RR.f0_axis(ds, RR.DATASET_PROFILE[ds].variant)


def base_clips(ds: str, spec_dir: str, emb: str) -> list[str]:
    """The clip folders present in the base layer (the list step 7 passes to f0 and the count its cell expects)."""
    d = P.layer_path("base", ds, hand=spec_dir, emb=emb)
    return sorted(c.name for c in d.glob("*") if c.is_dir())


def _layer(what: str, ds: str, spec_dir: str, emb: str, side: str) -> str:
    """Resolve a path token of a Run argv."""
    if what == "@base":
        return str(P.layer_path("base", ds, hand=spec_dir, emb=emb))
    if what == "@f0":
        return str(P.main_root() / f0_axis(ds)["shadow"])
    if what == "@clips":
        names = base_clips(ds, spec_dir, emb)
        cdir = P.run_path("run_log", P.hand_id(spec_dir), tool="hand_pipeline")
        cdir.mkdir(parents=True, exist_ok=True)
        f = cdir / f"clips_{ds}_{emb}.txt"
        f.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        return str(f)
    return str(P.wrist_base_dir(spec_dir, side, ds))


def f0_cells(ds: str, spec_dir: str) -> dict[str, dict]:
    """Step 7 per embodiment: {judged, need, run_failure, no_ledger, no_marker, first}. need = base_clips; judged =
    clips whose last state in this hand's f0 ledger is settled (``shadow_status.unsettled_reason``). Ledger lines carry
    the
    clip name only, so a clip name present in two embodiments is fatal."""
    from dexengine.placement.judge.f0_key import MARKER  # noqa: PLC0415
    from dexengine.retarget.f0 import shadow_status as _F0  # noqa: PLC0415

    root = P.main_root() / f0_axis(ds)["shadow"] / spec_dir
    want: dict[str, str] = {}
    for emb in T.embs(ds):
        for c in P.read_clip_list(base_clips(ds, spec_dir, emb), dataset=ds, quiet=True):
            if c in want:
                raise InputMissingError(
                    f"[hand_pipeline] {ds}: clip {c!r} is in both emb {want[c]} and {emb} -- the "
                    f"f0 ledger ({_F0.LEDGER_NAME}) names clips only, so step 7 cannot be counted"
                )
            want[c] = emb
    last = _F0.ledger_last(root / _F0.LEDGER_NAME, set(want))
    out = {
        e: {"judged": 0, "need": 0, "run_failure": 0, "no_ledger": 0, "no_marker": 0, "first": ""} for e in T.embs(ds)
    }
    for c, emb in want.items():
        st = last.get(c, (None, ""))[0]
        why = _F0.unsettled_reason(st, root / emb / c / MARKER)
        o = out[emb]
        o["need"] += 1
        if not why:
            o["judged"] += 1
            continue
        o[why] += 1
        o["first"] = o["first"] or f"{emb}/{c}:{st or 'no ledger'}"
    return out


def f0_gap_text(o: dict) -> str:
    """One line on the clips step 7 still has to settle (printed with its plan)."""
    miss = o["need"] - o["judged"]
    return (
        f"judged {o['judged']}/{o['need']} · missing clips {miss} (run failure {o['run_failure']} · no ledger "
        f"{o['no_ledger']} · no marker {o['no_marker']})" + (f" · first {o['first']}" if o["first"] else "")
    )


def _axes(cmd, ds: str, spec_dir: str) -> list:
    """(emb, side) pairs a step runs on: one empty pair, each embodiment with a base layer, or each of its sides."""
    if cmd.axis == "":
        axes = [("", "")]
    else:
        embs = [e for e in T.embs(ds) if any(P.layer_path("base", ds, hand=spec_dir, emb=e).glob("*"))]
        axes = [(e, "") for e in embs] if cmd.axis == "emb" else [(e, sd) for e in embs for sd in T.emb_sides(e)]
    return axes


def _f0_run(argv: list, ds: str, spec_dir: str, emb: str, side: str, f0_force: bool, f0c: dict) -> dict:
    """One step-7 (f0) run: its --f0-root must be the f0 axis root; `expect` carries the scene, root and gap the
    caller checks against the completion ledger."""
    # Root and scene = the axis the judge preparation reads; the f0 completion ledger records both and the
    # caller compares them with `expect`.
    ax = f0_axis(ds)
    shadow = str(P.main_root() / ax["shadow"])
    got = argv[argv.index("--f0-root") + 1]
    if got != shadow:
        raise SystemExit(f"[hand_pipeline] step 7 --f0-root differs from the f0 axis: {got} != {shadow}")
    if f0_force:
        argv.append("--force")
    return {
        "emb": emb,
        "side": side,
        "argv": argv,
        "expect": {
            "scene": ax["scene"],
            "shadow_root": f"{shadow}/{spec_dir}",
            "gap": f"emb={emb} " + f0_gap_text(f0c[emb]),
        },
    }


def _row(step: str) -> tuple:
    """The table row of a driven step (its id); an unknown id or a shell-only row is fatal."""
    st = [g for g in T.STAGES if T.step_id(g) == step]
    if len(st) != 1:
        raise SystemExit(
            f"[hand_pipeline] --emit-run {step!r} matches {len(st)} steps (known: {[g[0] for g in T.STAGES]})"
        )
    if not isinstance(st[0][3], T.Run):
        raise SystemExit(f"[hand_pipeline] '{st[0][0]}' is not driven here (shell entry point) -- run: {st[0][3]}")
    return st[0]


def _argv(step: str, cmd, src, py: str, ds: str, spec_dir: str, emb: str, side: str) -> list[str]:
    """One run's argv: the interpreter and the row's argv with slots and path tokens filled, behind the run guard (and
    the judge scene for a step that reads the desk)."""
    hand = RR.ACTIVE_HAND
    argv = [py] + [
        _layer(a, ds, spec_dir, emb, side)
        if a in T._DOC_TOKEN
        else str(T.REPO / a)
        if a.startswith("scripts/")
        else a.format(ds=ds, hand=hand, emb=emb, side=side)
        for a in src
    ]
    guard = [f"{P.STAGE1_STEP_ENV}={step}", f"{P.STAGE1_SIDECARS_ENV}={','.join(T.sidecars_ready(step))}"]
    argv = ["env", *guard, *([f"DEXENGINE_SCENE={f0_axis(ds)['scene']}"] if cmd.scene else []), *argv]
    return argv


def step_argv(step: str, ds: str, *, emb: str = "", side: str = "") -> list[str]:
    """One run of table step `step` with ``{ds}`` = `ds` (a comma list goes in as is) for the active hand: the argv
    owner for a caller that scopes the run itself and appends its own arguments (stage1.split_apply's piece runs)."""
    _name, _pat, slot, cmd = _row(step)
    py = T.py(T._PY_SLOT[slot]) if slot else "bash"
    return _argv(step, cmd, cmd.argv, py, ds, RR.hand_dir(), emb, side)


def emit_run(step: str, ds: str, *, f0_force: bool, reuse: bool = False) -> dict:
    """The argv lists that run step `step` (its id, e.g. ``2b``) on dataset `ds` for the active hand.

    Embodiments come from the dataset profile, restricted to those with a base layer. ``reuse`` sends the step's
    per-tree reuse command instead of its argv when it has one (the result's ``reuse`` names what was sent).
    Outside the step's rule scope the result has no runs and says ``not_applicable``.
    """
    name, pat, slot, cmd = _row(step)
    hand, spec_dir = RR.ACTIVE_HAND, RR.hand_dir()
    if not T.applies(cmd, ds):
        return {
            "stage": name,
            "dataset": ds,
            "hand": hand,
            "spec_dir": spec_dir,
            "f0_force": False,
            "reuse": [],
            "runs": [],
            "not_applicable": f"outside the scope {list(T.rule_scope(cmd))} of rule {cmd.rule}",
        }
    py = T.py(T._PY_SLOT[slot]) if slot else "bash"
    axes = _axes(cmd, ds, spec_dir)
    is_f0 = pat == T._F0_JUDGED  # the f0 step is the row with the f0 cell (as in table.split_steps)
    f0c = f0_cells(ds, spec_dir) if is_f0 else {}
    src = cmd.reuse if reuse and cmd.reuse else cmd.argv
    used = [Path(src[0]).name] + [a for a in src[1:] if a not in cmd.argv] if src is not cmd.argv else []
    runs = []
    for emb, side in axes:
        argv = _argv(step, cmd, src, py, ds, spec_dir, emb, side)
        if is_f0:
            runs.append(_f0_run(argv, ds, spec_dir, emb, side, f0_force, f0c))
            continue
        runs.append({"emb": emb, "side": side, "argv": argv})
    return {
        "stage": name,
        "dataset": ds,
        "hand": hand,
        "spec_dir": spec_dir,
        "f0_force": bool(f0_force and is_f0),
        "reuse": used,
        "runs": runs,
    }


def stage1_base_root(ds: str, spec_dir: str) -> Path:
    """The base-layer root of one hand (where the rig and prior stamps live). A root not named after the hand is a
    flat layout without a hand axis (a frozen generation's place); stage 1 refuses to write there."""
    root = P.layer_path("base", ds, hand=spec_dir, emb=T.embs(ds)[0]).parent
    if root.name != spec_dir:
        raise SystemExit(
            f"[STAGE1_BASE] ✗ {ds} × {spec_dir}: the base root {root} is not a hand folder -- stage 1 "
            "does not write into a layout without a hand axis. Remove this dataset from --datasets."
        )
    return root


def prior_file(spec_dir: str) -> Path:
    """The retargeting pose prior of a hand (in its rig folder, `paths.hand_rig_dir`). Its bytes are part of the
    stage-1 reuse identity."""
    return P.hand_rig_dir(spec_dir) / "retarget_prior.yaml"


def prior_identity(spec_dir: str) -> str:
    """The prior's identity: first 16 hex chars of the sha256 of the prior file. The rig hash covers geometry only, so
    without this a prior-only change would let stage 1 reuse old outputs."""
    f = prior_file(spec_dir)
    if not f.is_file():
        raise InputMissingError(f"[hand_pipeline] no retargeting prior: {f}")
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16]


def _stamp(f: Path) -> str | None:
    """First word of a stamp file (the prior stamp is `<identity>\\t<source>`); None when missing or empty."""
    w = f.read_text(encoding="utf-8").split() if f.is_file() else []
    return w[0] if w else None


def emit_rig(ds: str) -> dict:
    """The rig identity of a dataset's outputs for the active hand: ``registry`` (declared, `paths.hand_rig_sha`),
    ``live_rig`` (hash of the rig folder now), ``measured`` (``hand_asset_sha256_xml3`` in base task_info, the
    independent oracle), the base ``stamp``/``prior_stamp`` and the retarget stage's ``stage_stamp``/
    ``stage_prior_stamp``. Two different measured hashes in one base layer are fatal (mixed outputs)."""
    dl = P.data_layout
    spec_dir = RR.hand_dir()
    base_root = stage1_base_root(ds, spec_dir)
    measured, n = None, 0
    for f in sorted(base_root.glob("*/*/task_info.json"))[:64]:
        try:
            v = json.loads(f.read_text(encoding="utf-8")).get("hand_asset_sha256_xml3")
        except (OSError, ValueError):
            continue
        if v:
            n += 1
            if measured is None:
                measured = v
            elif measured != v:
                raise SystemExit(
                    f"[hand_pipeline] {ds}: two rig hashes in the base layer ({measured[:12]} · "
                    f"{v[:12]}) -- mixed outputs; decide by hand"
                )
    stamp_f = base_root / RIG_STAMP_NAME
    prior_f = base_root / PRIOR_STAMP_NAME
    stage = P.retarget_stage_root(RR.ACTIVE_HAND) / "processed" / ds
    rig_dir = P.hand_rig_dir(spec_dir)
    return {
        "dataset": ds,
        "spec_dir": spec_dir,
        "registry": P.hand_rig_sha(spec_dir),
        "live_rig": dl.asset_hash(str(rig_dir)) if rig_dir.is_dir() else None,
        "measured": measured,
        "n_measured": n,
        "stamp": _stamp(stamp_f),
        "stamp_path": str(stamp_f),
        "prior": prior_identity(spec_dir),
        "prior_stamp": _stamp(prior_f),
        "prior_stamp_path": str(prior_f),
        "stage_stamp": _stamp(stage / RIG_STAMP_NAME),
        "stage_prior_stamp": _stamp(stage / PRIOR_STAMP_NAME),
        "stage_dir": str(stage),
        "base_root": str(base_root),
    }


def invalidate_stage_stamps(ds: str) -> str:
    """Before retargeting writes into the stage with the current rig: remove the stage's rig and prior stamps when
    they differ from the current identity (a stamp says "solved and ingested with that rig and prior" and becomes
    false once another rig writes). Returns one ``[STAGE_STAMP] ...`` line; a failed removal is fatal."""
    r = emit_rig(ds)
    st = Path(r["stage_dir"])
    old, cur = (r["stage_stamp"], r["stage_prior_stamp"]), (r["registry"], r["prior"])
    if all(o is None or o == c for o, c in zip(old, cur, strict=True)):
        return (
            f"[STAGE_STAMP] {ds} stamps {(old[0] or '-')[:12]}/{(old[1] or '-')[:12]} = current identity or none "
            "-- kept"
        )
    for name in (RIG_STAMP_NAME, PRIOR_STAMP_NAME):
        (st / name).unlink(missing_ok=True)
        if (st / name).exists():
            raise SystemExit(f"[hand_pipeline] could not remove {st / name}")
    was = f"{(old[0] or '-')[:12]}/{(old[1] or '-')[:12]}"
    return (
        f"[STAGE_STAMP] {ds} old stamps {was} != current {cur[0][:12]}/{cur[1][:12]}"
        " -- invalidated (removed; the ingest stamps again)"
    )
