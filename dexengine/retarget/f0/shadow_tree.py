"""f0 shadow trees on disk: mirror a source clip into the shadow, refresh pass-through files, decide whether a shadow
is current, prepare a clip for the solve (prep) and write its result and marker (finish)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from dexengine.core import blas_pin
from dexengine.placement.judge.f0_key import MARKER
from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0.shadow_key import (
    FP_DESIGN,
    FP_DESIGN_V1,
    SCENE_XMLS,
    cache_verdict,
    clip_sides,
    input_key_diff,
    rewrite_scene_dirs,
    written_files,
)
from dexengine.retarget.f0.shadow_status import DATA_DEFECT, SIDES, SourceDefectError


def refresh_passthrough(clip: str, stage_src: Path, dst_tree: Path, src_layer: str) -> int:
    """On a cache hit, sync the shadow's pass-through files (mirrored but neither read nor written by f0:
    task_info.json, body_positions, retarget_diagnostics, scene xml) to the current base. f0 output does not change when
    files outside the key change, but judging and baking read these shadow files. Hard links are re-made only when the
    inodes differ; scene xml is written only when the rewritten text differs. Returns the number of files changed. f0
    outputs (npz in written_files), markers and wrist shadows are untouched."""
    src, d = stage_src / clip, dst_tree / clip
    written = written_files(src_layer)
    rewrite = set(SCENE_XMLS) if src_layer == "base" else set()
    n = 0
    for p in sorted(src.rglob("*")):
        if p.is_dir():
            continue
        rel = str(p.relative_to(src))
        out = d / rel
        if rel in rewrite:
            txt = rewrite_scene_dirs(p.read_text(), src)
            if not out.is_file() or out.read_text() != txt:
                tmp = out.with_name(out.name + ".pt_tmp")
                tmp.write_text(txt)
                os.replace(tmp, out)
                n += 1
            continue
        if rel in written:
            continue
        if out.exists() and out.stat().st_ino == p.stat().st_ino:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".pt_tmp")
        try:
            os.link(p, tmp)
        except OSError:
            shutil.copy2(p, tmp)
        os.replace(tmp, out)
        n += 1
    return n


def shadow_is_current(clip: str, stage_src: Path, wrist_dirs: dict, dst_tree: Path, src_layer: str) -> str:
    """Verify, without re-running f0, whether an already-built shadow **came from the current input**.

    It is for backfilling old markers (from before the convention) that have no fingerprint. **Both**
    conditions have to be checked:

      (a) staging — a file `mirror_clip` created with `os.link` shares its inode with the original.
          If the original is written anew (the generator always writes a new file) the inodes diverge.
          The 4 `written` files are overwritten by f0, so they are excluded from the comparison.
      (b) wrist cache — the canonical wrist cache must **equal what the staging `body_positions.npz`
          gives now** (owner placement.fk.wrist_poses.cache_state — a content comparison, not an
          mtime one: 2b and rebases rewrite the file without changing it, so mtime would call
          fresh caches stale).

    Returns: "" if current (the fingerprint may be stamped), otherwise the reason string.
    """
    src, d = stage_src / clip, dst_tree / clip
    written = written_files(src_layer)
    for pth in sorted(src.rglob("*")):
        if not pth.is_file():
            continue
        rel = str(pth.relative_to(src))
        q = d / rel
        if not q.exists():
            return f"shadow missing:{rel}"
        if rel in written:
            continue
        if pth.stat().st_ino != q.stat().st_ino:
            return f"staging rewritten:{rel}"
    bp = src / "0/body_positions.npz"
    if bp.exists():
        from dexengine.placement.fk import wrist_poses as WP  # noqa: PLC0415 — wrist-cache freshness owner

        for side in clip_sides(stage_src, clip):
            why = WP.cache_state(str(bp), wrist_dirs[side] / f"{clip}.npz", side)[0]
            if why == "absent":
                return f"wrist cache missing:{side}"
            if why:
                return f"wrist cache differs from body_positions ({why}):{side}"
    return ""


def _mirror_files(d, src, src_layer):
    """Hard-link every staging file into the shadow clip; the files repair writes are real copies and the base
    scene xml files are rewritten with absolute asset paths."""
    # The files f0.repair **writes** must be real copies (a hard link would pollute the original).
    written = written_files(src_layer)
    rewrite = set(SCENE_XMLS) if src_layer == "base" else set()
    for p in sorted(src.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = d / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if str(rel) in rewrite:
            out.write_text(rewrite_scene_dirs(p.read_text(), src))
        elif str(rel) in written:
            shutil.copy2(p, out)
        else:
            try:
                os.link(p, out)  # 0 space — read-only file
            except OSError:
                shutil.copy2(p, out)


def _drop_idle_wrist_shadows(_sides, clip, dst_wrist_dirs):
    """Delete wrist shadows an earlier run left for a side that does not manipulate in this tree."""
    for side in SIDES:
        if side in _sides:
            continue
        stale = dst_wrist_dirs[side] / f"{clip}.npz"
        if stale.exists():
            # A non-manipulating side's shadow left by an earlier run (a different role or generation) — left in place,
            # the judge reads a file on the old time axis. Delete it.
            stale.unlink()
            print(f"[f0_prejudge] {clip}: removed leftover wrist shadow of non-manipulating side {side}", flush=True)


def mirror_clip(
    clip: str, stage_src: Path, wrist_dirs: dict, dst_tree: Path, dst_wrist_dirs: dict, src_layer: str
) -> str | None:
    """Copy the staging and the wrist cache into the shadow. Only what is modified is a real copy, the
    rest are hard links. In the base layer the two scene xml files are real copies whose
    meshdir/texturedir have been rewritten as absolute paths (rewrite_scene_dirs)."""
    src = stage_src / clip
    kin = src / "0/trajectory_kinematic.npz"
    if not kin.exists():
        return f"no staging: {kin}"
    _sides = clip_sides(stage_src, clip)
    for side in _sides:
        _w = wrist_dirs[side] / f"{clip}.npz"
        if not _w.exists():
            # Always record which path was looked at. Leaving only "missing" sends the reader
            # digging through other directories.
            return f"no wrist cache: {side} ({_w})"
    d = dst_tree / clip
    if d.exists():
        shutil.rmtree(d)
    (d / "0").mkdir(parents=True, exist_ok=True)
    _mirror_files(d, src, src_layer)
    if src_layer == "base":
        for x in SCENE_XMLS:
            if not (d / x).exists():
                return f"the base tree lacks {x}: {src / x}"
    for side in _sides:
        wd = dst_wrist_dirs[side]
        wd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wrist_dirs[side] / f"{clip}.npz", wd / f"{clip}.npz")
    _drop_idle_wrist_shadows(_sides, clip, dst_wrist_dirs)
    return None


def _repair_argv(a, clip, dst_tree, stage_src):
    """The f0.repair argv of one clip and the sides it repairs (only the hands present in the tree)."""
    # f0.repair looks only at the shadow tree (--tree_root) and the wrist (--wrist_root); --dataset
    # is used for the object mesh and surface_points layer (layer_path("meshes", ds)) and for checking
    # task_info.dataset_name — it is threaded through as an argument, not as an env var.
    sides = clip_sides(stage_src, clip)
    argv = [
        "--hand",
        a.hand,
        "--dataset",
        a.dataset,
        "--in_place",
        "--side",
        ("both" if len(sides) == 2 else sides[0]),  # 2a: only the hands in the tree
        "--sequence",
        clip,
        "--tree_root",
        str(dst_tree),
        "--wrist_root",
        str(dst_tree.parent),
    ]
    if a.force:
        argv.append("--force")
    if a.side_budget_evals is not None:
        argv += ["--side_budget_evals", str(a.side_budget_evals)]
    return argv, sides


def prep(clip: str, a, stage_src: Path, wrist_dirs: dict, dst_tree: Path, dst_wrist_dirs: dict) -> dict:
    """Pre-run steps of one clip (cache decision, mirroring, argv). Returns a result dict (CACHED/FAIL_MIRROR) for a
    finished clip, otherwise a job dict (`job=True`); the GPU worker (`gpu --worker`) runs it and `finish` classifies
    it."""
    t0 = time.time()
    d = dst_tree / clip
    mark = d / MARKER
    stale = ""  # reason for regeneration (if set, the cache was not used)
    # The single owner of the cache decision = cache_verdict (v2 content key, strict v1 marker migration).
    # If the key cannot be measured (missing input file, I/O, declared data defect), stop here, before mirroring, so the
    # clip's shadow is untouched. `_guard_prep` classifies: only a declared data defect (SourceDefectError) is
    # DATA_DEFECT; everything else is a run failure (FAIL_EXC -> run rc != 0, stage2 [F0_RUN_FAIL]). Translating a key
    # failure into 'stale' and mirroring would let an environment error overwrite the canonical shadow.
    pk: dict = {}  # per-component input hashes (input_parts, STALE diff)
    verdict, why, key_now, j = cache_verdict(
        clip, stage_src, wrist_dirs, dst_tree, a.dataset, a.src_layer, parts_out=pk
    )
    if not a.force and verdict in ("FRESH", "MIGRATE"):
        npt = refresh_passthrough(clip, stage_src, dst_tree, a.src_layer)
        if verdict == "MIGRATE":  # write the v2 key into the marker (then FRESH)
            j = dict(
                j, input_key=key_now, input_parts=pk, fp_design=FP_DESIGN, migrated_from=FP_DESIGN_V1, migrate_why=why
            )
            tmp = mark.with_name(mark.name + ".tmp")
            tmp.write_text(json.dumps(j, ensure_ascii=False, indent=1))
            os.replace(tmp, mark)
        return {
            "clip": clip,
            "status": "CACHED",
            "P": int(j.get("P", 0) or 0),
            "t": 0.0,
            "log": str(j.get("f0_status", "")),
            "stale": "",
            "bskip": int(j.get("budget_skip", 0) or 0),
            "evals": str(j.get("budget_evals_used", "") or ""),
            "migrated": int(verdict == "MIGRATE"),
            "passthrough": npt,
        }
    if not a.force and verdict == "STALE":
        # **It is not silently reused** — a stale shadow would feed the judge pre-repair inputs.
        # with a different key, append what changed (components, task_info fields) to the reason (record only)
        stale = f"{why}({input_key_diff((j or {}).get('input_parts'), pk)})" if why == "input_changed" else why
    if stale:
        print(f"  STALE→regenerate [{stale}] {clip}", flush=True)
    err = mirror_clip(clip, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs, a.src_layer)
    if err:
        return {"clip": clip, "status": "FAIL_MIRROR", "P": 0, "t": time.time() - t0, "log": err, "stale": stale}
    argv, sides = _repair_argv(a, clip, dst_tree, stage_src)
    return {
        "job": True,
        "clip": clip,
        "t0": t0,
        "d": d,
        "mark": mark,
        "key_now": key_now,
        "stale": stale,
        "sides": sides,
        "argv": argv,
        "input_parts": pk,
    }


def applied_scene() -> str:
    """The scene applied in this process.

    Owned by coords.scene_mode (env DEXENGINE_SCENE); the rule is not restated here."""
    from dexengine.placement.coords import core as _coords  # noqa: PLC0415

    return _coords.scene_mode()


def _compute_env() -> dict:
    """The run's compute environment (blas_pin.compute_env), read once per process."""
    global _ENV
    if _ENV is None:
        _ENV = blas_pin.compute_env()
    return _ENV


_ENV = None


def _write_marker(P, a, bskip, clip, d, evals, job, key_now, mode, stage_src, status, wrist_dirs):
    """Write the shadow's marker (status, input key, budget use, applied scene) next to the repaired tree."""
    (d / MARKER).write_text(
        json.dumps(
            {
                "clip": clip,
                "hand": a.hand,
                "status": status,
                "P": int(P),
                "f0_status": mode,
                "dataset": a.dataset,
                "emb": a.emb,
                "src_layer": a.src_layer,
                "src_stage": str(stage_src / clip),
                "src_wrist": str(wrist_dirs["right"].parent),
                # f0 input key (v2): the basis on which the next run decides whether this shadow came from the current
                # inputs (cache_verdict). Empty when the key could not be measured, so it is never reused
                # (conservative).
                "input_key": key_now,
                "fp_design": FP_DESIGN,
                # per-component key hashes, so the next STALE reason says what changed; not used for the decision
                "input_parts": job.get("input_parts") or {},
                "budget_skip": int(bskip),
                "budget_evals_used": evals,  # under which budget this shadow was made
                # scene that set this shadow's desk plane (record only; outside the cache key)
                "scene": applied_scene(),
                "design": "f0_before_judge_v1(C109)",
                # where and with what this shadow was computed (host · CPU · package versions · loaded BLAS kernel);
                # a record only — outside the cache key (input_key) and the identity key (f0_key.KEY_FIELDS)
                "compute_env": _compute_env(),
            },
            ensure_ascii=False,
            indent=1,
        )
    )


def _no_budget_mark_row(clip, t0, bud, sides, out, log, stale):
    """The FAIL_NO_BUDGET_MARK row: the [F0_BUDGET] line count differs from the side count (run invalid)."""
    # No marker line + f0.repair's ALREADY_REPAIRED line = another f0 run wrote the repair
    # ledger (F0_REPAIR_INPLACE.json) into the shadow this run had just mirrored (mirror_clip deletes and rebuilds
    # the shadow, so it cannot be this run's). Two prejudges solving the same shadow root raced: the status stays
    # (run invalid) and only the cause is recorded.
    _race = not bud and PH.ALREADY_REPAIRED in out
    return {
        "clip": clip,
        "status": "FAIL_NO_BUDGET_MARK",
        "P": 0,
        "t": time.time() - t0,
        "log": (
            f"[F0_BUDGET] marker lines {len(bud)} ≠ sides {len(sides)} "
            f"— run invalid (one budget line per side) · "
            + ("[F0_RACE] another f0 run wrote the repair ledger into the same shadow first · " if _race else "")
            + log[-200:]
        ),
        "stale": stale,
        "bskip": 0,
        "evals": "",
    }


def finish(job: dict, rc: int, out: str, a, stage_src: Path, wrist_dirs: dict) -> dict:
    """Worker output (rc, out) -> status classification + marker. The single owner of the classification; its input is
    the output text of repair.main_steps, classified by the outcome tokens of f0.phrases."""
    clip, t0, d, key_now, stale, sides = (job[k] for k in ("clip", "t0", "d", "key_now", "stale", "sides"))
    log = out.replace("\n", " | ")[-400:]
    if rc != 0:
        return {"clip": clip, "status": "FAIL_F0", "P": 0, "t": time.time() - t0, "log": log, "stale": stale}
    # Per-side budget marker. f0.repair prints exactly one line per side:
    #   [F0_BUDGET] side=<side> evals=<used>/<budget> status=OK|EXHAUSTED
    # A line count different from the side count invalidates the run: completion is declared only on matching counts.
    # This keeps a partial completion from looking like success at the ledger level.
    bud = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[F0_BUDGET]")]
    if len(bud) != len(sides):
        return _no_budget_mark_row(clip, t0, bud, sides, out, log, stale)
    bskip = sum(1 for ln in bud if "status=EXHAUSTED" in ln)
    evals = ",".join(ln.split("side=", 1)[1].replace(" evals=", "=").split(" status=")[0] for ln in bud)
    # **All three branches are recorded in the ledger** (to tell "it never ran" from "it ran and there
    # was nothing to do").
    #   REPAIRED  repaired → F0_REPAIR_INPLACE.json (P>0). Length T→T+P.
    #   CLEAN     n_pen=0, so no repair needed → dexengine.retarget.f0.repair records F0_REPAIR.json (P=0).
    #   SHALLOW   there is penetration, but the max depth ≤ DEXENGINE_F0_SKIP_DEPTH_MM (10mm by
    #             default), so the repair is skipped (dexengine.retarget.f0.repair's SKIP). It leaves no ledger, so
    #             one is left here — P=0, arrays unchanged. The old chain gave the same result (no pad).
    P, mode = 0, ""
    if (d / "F0_REPAIR_INPLACE.json").exists():
        j = json.loads((d / "F0_REPAIR_INPLACE.json").read_text())
        P, mode = int(j.get("P", 0)), str(j.get("mode", ""))
        status = "REPAIRED" if P else "CLEAN"
    elif (d / "F0_REPAIR.json").exists():
        j = json.loads((d / "F0_REPAIR.json").read_text())
        P = int(j.get("n_blend_effective", 0) or 0)
        status = "REPAIRED" if P else "CLEAN"
    elif PH.SKIP_BUDGET in out:
        # A side abandoned on budget is not a shallow penetration: checking `"SKIP" in out` first would classify this
        # clip as SHALLOW (= success set), although such a side can be far deeper than the shallow threshold. The
        # trigger is "zero repaired sides + a SKIP phrase in the output" (the two branches above look at the repair json
        # first). The clip goes to REPAIR_FAIL (it enters judging and the gate decides).
        status, mode = "REPAIR_FAIL", f"f0 budget SKIP {bskip} side(s) (hold pad kept) — {evals}"
    elif PH.SKIP in out:
        status, mode = "SHALLOW", "f0 SKIP (shallow penetration) — no pad"
    elif PH.PAD_TOO_BIG in out or PH.INPLACE_FAILED in out:
        # The branch where f0.repair **refused normally** (required padding > the cap). rc=0 and no
        # ledger — without a cause classification (it is not FAIL_NOLEDGER, 'unexpected') an upstream
        # problem such as stale input is mistaken for an f0 bug.
        # MARKER is written before returning: the shadow copy exists (mirror_clip above) and
        # JUDGEABLE_STATUSES says these clips may enter judging (the gate decides) — stage2's preflight
        # looks for MARKER, so without it they are dropped as "no f0 shadow" and fall to MISSING in the
        # ledger. The status goes into the marker and the clip goes on; the refusal reason stays in the log.
        status, mode = "REFUSED_BIG", "f0 refused (over-large repair) — original kept · " + log[-200:]
    elif PH.OVERLAP_LEFT in out or PH.UNREPAIRED in out:
        # The branch where f0.repair could not fix it **and said why**: within the hard constraints
        # (arm-to-torso clearance and the like) it could not bring the first-frame overlap to 0. It ends
        # with rc=0 while keeping the hold pad and leaves no ledger — it is not FAIL_NOLEDGER
        # ("unexpected"). The same reason REFUSED_BIG is split out: **calling a failure that has a cause
        # 'unexpected' makes an upstream problem look like an f0 bug**. The batch keeps going (the hold
        # pad stays) — the gate makes the final call.
        _why = next((ln.strip() for ln in out.splitlines() if PH.OVERLAP_LEFT in ln), PH.OVERLAP_LEFT)
        status, mode = (
            "REPAIR_FAIL",
            "f0 not repairable (hold pad kept) — " + _why[:200],
        )  # marker, then on (same as REFUSED_BIG)
    else:
        # No repair, not a SKIP either, and no ledger = unexpected. It is not passed over silently.
        return {"clip": clip, "status": "FAIL_NOLEDGER", "P": 0, "t": time.time() - t0, "log": log, "stale": stale}
    _write_marker(P, a, bskip, clip, d, evals, job, key_now, mode, stage_src, status, wrist_dirs)
    return {
        "clip": clip,
        "status": status,
        "P": P,
        "t": time.time() - t0,
        "log": mode,
        "stale": stale,
        "bskip": bskip,
        "evals": evals,
    }


def _guard_prep(c, a, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs):
    try:
        return prep(c, a, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs)
    except SourceDefectError as e:  # declared data defect; only this is DATA_DEFECT
        return {
            "clip": c,
            "status": DATA_DEFECT,
            "P": 0,
            "t": 0.0,
            "log": f"{type(e).__name__}: {e}"[:400],
            "stale": "",
        }
    except Exception as e:  # noqa: BLE001 — anything else: run failure
        return {"clip": c, "status": "FAIL_EXC", "P": 0, "t": 0.0, "log": f"{type(e).__name__}: {e}"[:400], "stale": ""}
