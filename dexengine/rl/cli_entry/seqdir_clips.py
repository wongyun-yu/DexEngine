"""Per-clip steps of the trainer's --sequence_dir resolution (cli_entry.seqdir._resolve_sequence_dir): the per-clip
artifact scan, the articulation / missing-artifact checks, the run-group flags written to args_cli and the
contact-sidecar check."""

from __future__ import annotations

import os
import sys


def contact_mano_check(parser, args_cli, root: str, trees: list[str]) -> None:
    """Count the trees without ``contact_mano_<side>.npz``; refuse them only with DEXENGINE_REQUIRE_CONTACT_MANO=1.

    Nothing at runtime reads the file (it exists only for clips with an upstream MANO export), so by default a
    missing file is reported, not fatal.
    """
    side = "left" if getattr(args_cli, "left_uni", 0) else "right"
    missing = [t for t in trees if not os.path.exists(os.path.join(root, t, "0", f"contact_mano_{side}.npz"))]
    if missing and os.environ.get("DEXENGINE_REQUIRE_CONTACT_MANO", "0") == "1":
        parser.error(
            f"contact_mano_{side}.npz missing in {len(missing)}/{len(trees)} clips"
            f"(DEXENGINE_REQUIRE_CONTACT_MANO=1) — {root}:\n  " + "\n  ".join(missing)
        )
    if missing:
        print(
            f"[contact_mano] contact_mano_{side}.npz missing in {len(missing)}/{len(trees)} clips "
            f"— not used at runtime, so passed (set DEXENGINE_REQUIRE_CONTACT_MANO=1 to require it)"
        )


def collect_per_clip(
    _ql_mod,
    _sd,
    _sd_artic_q,
    _sd_bi,
    _sd_desk,
    _sd_desk_absent,
    _sd_discs,
    _sd_groups,
    _sd_ids,
    _sd_ids_left,
    _sd_ik_dir,
    _sd_ik_dir_L,
    _sd_missing,
    _sd_need_ik,
    _sd_paths,
    _sd_seqs,
    _sd_side,
    _sd_src,
    _sd_wrist_dir,
    _sd_wrist_dir_L,
    _tp_mod,
    _tru,
):
    """Per-clip object ids, run groups, identity sources, support discs, desk heights and articulation statements."""
    for _s in _sd_seqs:
        _ti_p = os.path.join(_sd, _s, "task_info.json")
        if not os.path.isfile(_ti_p):
            _sd_missing.append(f"{_s}: task_info.json")
            continue
        # Identity comes from the layout reader (objects[] role=main), not from a legacy key's basename.
        try:
            _lay = _ql_mod.read(os.path.join(_sd, _s))
        except Exception as _e:  # noqa: BLE001 — every unreadable clip is collected and reported together
            _sd_missing.append(f"{_s}: task_info objects/layout unreadable — {type(_e).__name__}: {_e}")
            continue
        _sd_src.add(_lay.identity_source)
        _sd_groups.append(_tru.run_group(_lay))
        _idle = _lay.idle_sides() if _sd_bi else ()
        if len(_idle) > 1:
            _sd_missing.append(f"{_s}: both sides idle (D16) — no manipulating hand (not a training target)")
            continue
        _idle = _idle[0] if _idle else ""
        _oid = _lay.main_object_id(_sd_side)
        if _oid is None and _idle == _sd_side:
            # The idle side has no object. Its object slot carries the ACTIVE side's object as a placeholder so every
            # bimanual registration path stays as it is; env_setup's apply_idle_side removes that prim and every
            # term bound to it before the env is built — the placeholder is never spawned or read.
            _oid = _lay.main_object_id("left")
        if _oid is None:
            _sd_missing.append(f"{_s}: {_sd_side} main object missing (objects[] role=main)")
            continue
        # Asset id (key of the usd / meshes paths) = object_asset_id(main object, articulation); a rigid object
        # keeps its id. An articulated object must carry the committed canonical joint (checked after the loop).
        _artR = _lay.main_articulation(_sd_side if _idle != _sd_side else "left")
        if _artR:
            _sd_artic_q.append((_s, _oid, _artR))
        _oid = _tp_mod.object_asset_id(_oid, _artR)
        _sd_ids.append(_oid)
        _note_clip_supports(_s, _lay, _sd_discs, _sd_desk, _sd_desk_absent)
        if _sd_bi:
            # Left partner: object asset and left-arm wrist/IK caches, preflighted like the active side.
            _lid = _lay.main_object_id("left") if _idle != "left" else _lay.main_object_id("right")  # idle: placeholder
            if _lid is None:
                _sd_missing.append(f"{_s}: left main object missing (objects[] role=main)")
                continue
            _artL = _lay.main_articulation("left" if _idle != "left" else "right")
            if _artL:
                _sd_artic_q.append((f"{_s}[left]", _lid, _artL))
            _lid = _tp_mod.object_asset_id(_lid, _artL)
            _sd_ids_left.append(_lid)
            _note_missing_left_inputs(
                _s, _sd_missing, _sd_paths, _sd_wrist_dir_L, _sd_ik_dir_L, _sd_need_ik, _lid, _artL, _idle
            )
        _note_missing_clip_inputs(
            _sd_ik_dir, _sd_missing, _sd_need_ik, _sd_paths, _sd_side, _sd_wrist_dir, _artR, _idle, _oid, _s
        )


def _note_clip_supports(_s, _lay, _sd_discs, _sd_desk, _sd_desk_absent):
    """Record the clip's support discs: the aerial discs, or the desk-support discs (a clip without one is noted)."""
    if getattr(_lay, "scene", "table") == "aerial":
        _sd_discs[_s] = list((getattr(_lay, "aerial", None) or {}).get("discs") or [])
    else:
        _dsup = getattr(_lay, "desk_support", None)
        if _dsup is None:
            _sd_desk_absent.append(_s)
        _sd_desk[_s] = list((_dsup or {}).get("discs") or [])


def _note_missing_left_inputs(
    _s, _sd_missing, _sd_paths, _sd_wrist_dir_L, _sd_ik_dir_L, _sd_need_ik, _lid, _artL, _idle
):
    """Bimanual: append to the missing list every left-partner input (object files unless idle, wrist and IK caches)."""
    # The idle side's object files are the placeholder's = the active side's, checked below/above once; the
    # idle arm's wrist and IK caches are still required (baked at the idle rest pose).
    for _reqL in (
        (
            (_sd_paths.usd_file(_lid, articulated=bool(_artL)), _sd_paths.surface_points_file(_lid))
            if _idle != "left"
            else ()
        )
        + (
            (
                os.path.join(_sd_wrist_dir_L, f"{_s}.npz")
                if _sd_wrist_dir_L
                else _sd_paths.precompute_file("wrist_poses", _s, "left")
            ),
        )
        + (
            (
                (
                    os.path.join(_sd_ik_dir_L, f"{_s}.npz")
                    if _sd_ik_dir_L
                    else _sd_paths.precompute_file("arm_ik_poses", _s, "left")
                ),
            )
            if _sd_need_ik
            else ()
        )
    ):
        if not os.path.exists(_reqL):
            _sd_missing.append(f"{_s}[left]: {_reqL}")


def _note_missing_clip_inputs(
    _sd_ik_dir, _sd_missing, _sd_need_ik, _sd_paths, _sd_side, _sd_wrist_dir, _artR, _idle, _oid, _s
):
    """Append to the missing list every per-clip input file (USD, surface points, precompute) that does not exist."""
    for _req in (
        (
            (_sd_paths.usd_file(_oid, articulated=bool(_artR)), _sd_paths.surface_points_file(_oid))
            if _idle != _sd_side
            else ()
        )
        + (
            (
                os.path.join(_sd_wrist_dir, f"{_s}.npz")
                if _sd_wrist_dir
                else _sd_paths.precompute_file("wrist_poses", _s, _sd_side)
            ),
        )
        + (
            (
                (
                    os.path.join(_sd_ik_dir, f"{_s}.npz")
                    if _sd_ik_dir
                    else _sd_paths.precompute_file("arm_ik_poses", _s, _sd_side)
                ),
            )
            if _sd_need_ik
            else ()
        )
    ):
        if not os.path.exists(_req):
            _sd_missing.append(f"{_s}: {_req}")


def apply_run_group(
    parser,
    args_cli,
    _roster,
    _sd,
    _sd_bi,
    _sd_desk,
    _sd_desk_absent,
    _sd_discs,
    _sd_groups,
    _sd_ids,
    _sd_ids_left,
    _sd_paths,
    _sd_seqs,
    _sd_side,
    _sd_src,
    _tru,
):
    """Run-group flags (idle side, shared object, articulated / aerial scene), the [OBJECTS] line and the args_cli
    fields the env build reads."""
    _idle_run, _ids_str, _right_txt, _sd_art_run, _sd_art_run_L, _sd_is_shared = _check_run_homogeneity(
        parser,
        args_cli,
        _roster,
        _sd_bi,
        _sd_desk,
        _sd_desk_absent,
        _sd_discs,
        _sd_groups,
        _sd_ids,
        _sd_ids_left,
        _sd_seqs,
        _sd_side,
        _tru,
    )
    _left_txt = (
        ("none(idle)" if _idle_run == "left" else "same" if _sd_is_shared else _ids_str(_sd_ids_left))
        if _sd_bi
        else "none"
    )
    print(
        f"[OBJECTS] right={_right_txt} "
        f"left={_left_txt} "
        f"shared={int(_sd_is_shared)} source={'|'.join(sorted(_sd_src))} clips={len(_sd_seqs)}",
        flush=True,
    )

    args_cli.data_dir = _sd
    args_cli.sequence = ",".join(_sd_seqs)
    setattr(
        args_cli, f"object_usd_{_sd_side}", ",".join(_sd_paths.usd_file(i, articulated=_sd_art_run) for i in _sd_ids)
    )
    setattr(args_cli, f"object_mesh_dir_{_sd_side}", ",".join(_sd_paths.mesh_dir(i) for i in _sd_ids))
    if _sd_bi:
        args_cli.object_usd_left = ",".join(_sd_paths.usd_file(i, articulated=_sd_art_run_L) for i in _sd_ids_left)
        args_cli.object_mesh_dir_left = ",".join(_sd_paths.mesh_dir(i) for i in _sd_ids_left)
    _apply_multi_object_switch(
        parser, args_cli, _sd, _sd_bi, _sd_ids, _sd_ids_left, _sd_paths, _sd_seqs, _sd_side, _sd_art_run, _sd_art_run_L
    )


def _apply_multi_object_switch(
    parser, args_cli, _sd, _sd_bi, _sd_ids, _sd_ids_left, _sd_paths, _sd_seqs, _sd_side, _sd_art_run, _sd_art_run_L
):
    """multi_object is a whole-run switch: set it (and the object paths) when either side varies per clip."""
    # multi_object is a whole-run switch: if either side varies per clip the run is multi-object, and a side that
    # does not vary stays a comma-list of the repeated value with the same length (keeps the lists aligned).
    _sd_vary = len(set(_sd_ids)) > 1 or (_sd_bi and len(set(_sd_ids_left)) > 1)
    # A bimanual sequence_dir run always takes the multi-clip env contract (static clip assignment +
    # replicate_physics=False + MultiAssetSpawner), whatever its clip and object counts: the replicated-physics
    # path was never validated for the bimanual scene, and eval, render and training must not change physics
    # contract with the clip count. Uni runs keep the single-clip contract.
    if _sd_bi and not _sd_vary:
        print(
            f"[sequence_dir] bi seqdir with {len(_sd_seqs)} clip(s), single object set -> "
            "multi-clip env contract forced (static clip assignment, replicate_physics=False)"
        )
        _sd_vary = True
    if _sd_vary and not getattr(args_cli, "multi_object", 0):
        args_cli.multi_object = 1
        print(
            f"[sequence_dir] distinct objects {_sd_side[0].upper()}={len(set(_sd_ids))}"
            + (f" L={len(set(_sd_ids_left))}" if _sd_bi else "")
            + " -> --multi_object 1 auto-enabled"
        )
    elif not _sd_vary:
        # single shared object: plain multi-clip path; collapse the comma-lists
        setattr(args_cli, f"object_usd_{_sd_side}", _sd_paths.usd_file(_sd_ids[0], articulated=_sd_art_run))
        setattr(args_cli, f"object_mesh_dir_{_sd_side}", _sd_paths.mesh_dir(_sd_ids[0]))
        if _sd_bi:
            args_cli.object_usd_left = _sd_paths.usd_file(_sd_ids_left[0], articulated=_sd_art_run_L)
            args_cli.object_mesh_dir_left = _sd_paths.mesh_dir(_sd_ids_left[0])
    print(f"[sequence_dir] {_sd}: {len(_sd_seqs)} sequences, side={_sd_side}, objects={sorted(set(_sd_ids))}")
    contact_mano_check(parser, args_cli, _sd, _sd_seqs)


def _check_run_homogeneity(
    parser,
    args_cli,
    _roster,
    _sd_bi,
    _sd_desk,
    _sd_desk_absent,
    _sd_discs,
    _sd_groups,
    _sd_ids,
    _sd_ids_left,
    _sd_seqs,
    _sd_side,
    _tru,
):
    """One run = one run key: the run flags of the group, checked homogeneous over its clips."""
    _sd_aer_run, _sd_art_run, _sd_art_run_L, _sd_is_shared = _apply_run_flags(
        parser, args_cli, _roster, _sd_bi, _sd_desk, _sd_discs, _sd_groups, _sd_seqs, _sd_side, _tru
    )
    if not _sd_aer_run and _sd_desk:
        print(
            f"[DESK_SUPPORT] sequence_dir: desk support stamped "
            f"{len(_sd_desk) - len(_sd_desk_absent)}/{len(_sd_desk)} clips · "
            f"with supports {sum(1 for v in _sd_desk.values() if v)} · "
            f"no stamp (old bake generation → 0 supports) {len(_sd_desk_absent)}"
            + (f" e.g. {_sd_desk_absent[0]}" if _sd_desk_absent else ""),
            flush=True,
        )
    if _sd_art_run or _sd_art_run_L or _sd_aer_run:
        print(
            f"[sequence_dir] D1/D20 run attributes: articulated {_sd_side}={int(_sd_art_run)}"
            + (f" left={int(_sd_art_run_L)}" if _sd_bi else "")
            + f" · scene={'aerial' if _sd_aer_run else 'table'}"
            + (f" · clips with discs {sum(1 for v in _sd_discs.values() if v)}/{len(_sd_seqs)}" if _sd_aer_run else ""),
            flush=True,
        )
    if _sd_is_shared and _sd_bi and any(a != b for a, b in zip(_sd_ids, _sd_ids_left, strict=True)):
        parser.error(
            "shared-object run with clips whose left/right main object ids differ — the reader's "
            "shared_object statement contradicts objects[]"
        )

    def _ids_str(_ids):
        _u = sorted(set(_ids))
        return _u[0] if len(_u) == 1 else f"{len(_u)}ids{_u[:4]}{'…' if len(_u) > 4 else ''}"

    _idle_run = getattr(args_cli, "idle_side", "")
    _right_txt = "none(idle)" if _idle_run == "right" else _ids_str(_sd_ids)
    return _idle_run, _ids_str, _right_txt, _sd_art_run, _sd_art_run_L, _sd_is_shared


def _apply_run_flags(parser, args_cli, _roster, _sd_bi, _sd_desk, _sd_discs, _sd_groups, _sd_seqs, _sd_side, _tru):
    """Apply the run key's flags (shared object, idle side, articulation, aerial scene, discs) to args_cli."""
    # Run homogeneity: one run = one run key (train_ready_util.one_run). Shared object, idle side, articulated and
    # aerial scene change the prim count and kind per run, so a mixed roster is refused; the run attributes are
    # derived from that key's group (train_ready_util.run_flags).
    try:
        _rk, _rg = _tru.one_run(_roster["axis"], _sd_seqs, _sd_groups)
    except ValueError as _e:
        parser.error(
            f"--sequence_dir: {_e}. Split the roster by run key (the train-set selector "
            f"dexengine/placement/train_set/select.py bundles by the same key) and launch each part "
            f"with DEXENGINE_SEQ_LIST"
        )
    _rf = _tru.run_flags(_rg, _sd_side, _sd_bi)
    print(f"[RUN_GROUP] key={_rk} clips={len(_sd_seqs)}", flush=True)
    if _sd_bi and _rf["shared"] and int(getattr(args_cli, "shared_object", 0) or 0) == 0:
        args_cli.shared_object = 1
    elif int(getattr(args_cli, "shared_object", 0) or 0) and not _sd_bi:
        parser.error("--shared_object 1 only applies with --bimanual 1")
    _sd_is_shared = bool(int(getattr(args_cli, "shared_object", 0) or 0))
    # Idle side: its prim and every term bound to it are removed from the scene cfg (env_setup apply_idle_side).
    args_cli.idle_side = _rf["idle_side"] if _sd_bi else ""
    if args_cli.idle_side and _sd_is_shared:
        parser.error(
            "an idle side and a shared object are declared in one run — a shared object belongs to clips "
            "where both hands are active (contradiction)"
        )
    if args_cli.idle_side:
        print(
            f"[IDLE_SIDE] idle side = {args_cli.idle_side} · clips {len(_sd_seqs)}/{len(_sd_seqs)} — "
            f"that hand holds its rest pose (owned by the tree and IK cache) · no object prim, object reward or "
            f"termination · object observations 0 (applied by dexengine.rl.env_setup apply_idle_side)",
            flush=True,
        )
    _sd_art_run, _sd_art_run_L, _sd_aer_run = _rf["art_active"], _rf["art_left"], _rf["aerial"]
    setattr(args_cli, f"object_articulated_{_sd_side}", int(_sd_art_run))
    if _sd_bi:
        args_cli.object_articulated_left = int(_sd_art_run if _sd_is_shared else _sd_art_run_L)
    args_cli.scene_aerial = int(_sd_aer_run)
    args_cli.aerial_discs = _sd_discs
    args_cli.desk_discs = _sd_desk
    return _sd_aer_run, _sd_art_run, _sd_art_run_L, _sd_is_shared


def check_artic_and_missing(parser, _sd_artic_q, _sd_ds, _sd_missing, _sd_side):
    """Articulated-object canon checks over the queued clips, then the missing-artifact failure."""
    if _sd_artic_q:
        import json as _json  # stdlib
        import subprocess as _subp  # stdlib

        from dexengine.core.repo import REPO_ROOT as _repo_root

        _ar_cli = str(_repo_root / "dexengine" / "data" / "artic" / "ranges.py")
        _ar_run = _subp.run(
            [sys.executable, _ar_cli, "--not-canonical-stdin", _sd_ds],
            capture_output=True,
            text=True,
            input=_json.dumps([[_o, _a] for _, _o, _a in _sd_artic_q]),
            timeout=600,
        )
        _ar_why = None
        if _ar_run.returncode == 0:
            try:
                _ar_why = _json.loads(_ar_run.stdout.strip().splitlines()[-1])
            except (IndexError, ValueError):
                _ar_why = None
        if not isinstance(_ar_why, list) or len(_ar_why) != len(_sd_artic_q):
            _sd_missing.append(
                f"the articulation canon check gave no answer for {len(_sd_artic_q)} queries "
                f"(rc {_ar_run.returncode}): " + (_ar_run.stderr.strip() or _ar_run.stdout.strip())[-1500:]
            )
        else:
            _sd_missing += [f"{_l}: {_w}" for (_l, _, _), _w in zip(_sd_artic_q, _ar_why, strict=True) if _w]
            print(
                f"[ARTIC_CANON] articulated {len(_sd_artic_q)} · not canonical {sum(map(bool, _ar_why))} "
                f"(dexengine.data.artic.ranges.not_canonical · child process)",
                flush=True,
            )
    if _sd_missing:
        parser.error(
            "--sequence_dir preflight FAILED (run scripts/stage2_ground.sh / "
            "its arm IK stage, side " + _sd_side + ", for the missing pieces):\n  " + "\n  ".join(_sd_missing)
        )
