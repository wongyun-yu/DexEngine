"""Per-finger contact audit of the rendered rollout (``DEXENGINE_CONTACT_AUDIT=1``).

Right hand only. Logs object-on-finger forces, net contact forces and fingertip-to-surface distances on the same
rollout that is saved to video, prints a per-finger summary and saves the per-frame arrays to
``/tmp/exp_audit_rich.npz``. Run it with ``--videos_warmup_attempts 0 --videos_max_attempts 1`` so the saved video is
the audited rollout and logged step N is saved frame N.
"""

from __future__ import annotations

import os
import sys

import torch

from dexengine.rl.swallow import note_once


def _contact_audit(raw_env, args_cli):
    """Audit state for the rollout loop: (on, rows, rich rows, finger names, log(frame_idx)).

    The last three are None when the audit is off.
    """
    _AUDIT = os.environ.get("DEXENGINE_CONTACT_AUDIT") == "1"
    _aud_rows = []
    _ar_rich = _AR_NAMES = _ar_log = None
    if _AUDIT:
        from dexengine.rl.tasks.mdp.rewards.contact_base import OBJ_SENSOR_NAMES as _AR_OBJN
        from dexengine.rl.tasks.mdp.rewards.contact_base import load_object_surface_points as _ar_surf_l
        from dexengine.rl.tasks.mdp.rewards.contact_base import resolve_tip_body_ids as _ar_tips
        from dexengine.rl.tasks.mdp.rewards.contact_base import tip_to_surface_dist as _ar_tip_dist

        _ar_side = "right"
        _ar_hand = raw_env.scene["right_hand"] if "right_hand" in raw_env.scene.keys() else raw_env.scene["robot"]  # noqa: SIM118  — InteractiveScene has no __contains__
        _ar_key = "object_right" if "object_right" in raw_env.scene.keys() else "object_left"  # noqa: SIM118  — InteractiveScene has no __contains__
        _ar_obj = raw_env.scene[_ar_key]
        _AR_NAMES, _ar_log, _ar_rich = _contact_audit_setup(
            raw_env,
            args_cli,
            _AR_OBJN,
            _ar_hand,
            _ar_obj,
            _ar_rich,
            _ar_side,
            _ar_surf_l,
            _ar_tip_dist,
            _ar_tips,
            _aud_rows,
            _AR_NAMES,
            _ar_log,
        )

    return _AUDIT, _aud_rows, _ar_rich, _AR_NAMES, _ar_log


def _contact_audit_setup(
    raw_env,
    args_cli,
    _AR_OBJN,
    _ar_hand,
    _ar_obj,
    _ar_rich,
    _ar_side,
    _ar_surf_l,
    _ar_tip_dist,
    _ar_tips,
    _aud_rows,
    _AR_NAMES,
    _ar_log,
):
    """Fingertip ids, object surfaces and logs of the per-step contact audit."""
    _ar_tip_ids = _ar_tips(_ar_hand, _ar_side)
    _ar_surf = _ar_surf_l(args_cli.object_mesh_dir, raw_env.device)
    _ar_objs = [raw_env.scene[n] for n in _AR_OBJN[_ar_side]]
    _ar_cs = raw_env.scene["right_hand_contact"]
    _ar_hbn = _ar_hand.data.body_names
    _ar_sbn = _ar_cs.body_names
    _ar_stids = [_ar_sbn.index(_ar_hbn[i]) if _ar_hbn[i] in _ar_sbn else -1 for i in _ar_tip_ids]
    _AR_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
    _ar_log, _ar_rich = _contact_audit_logs(
        raw_env,
        _ar_hand,
        _ar_obj,
        _ar_rich,
        _ar_tip_dist,
        _aud_rows,
        _ar_cs,
        _ar_objs,
        _ar_stids,
        _ar_surf,
        _ar_tip_ids,
        _ar_log,
    )

    return _AR_NAMES, _ar_log, _ar_rich


def _contact_audit_logs(
    raw_env,
    _ar_hand,
    _ar_obj,
    _ar_rich,
    _ar_tip_dist,
    _aud_rows,
    _ar_cs,
    _ar_objs,
    _ar_stids,
    _ar_surf,
    _ar_tip_ids,
    _ar_log,
):
    """Per-step contact audit log and its detailed rows."""
    _ar_origins = raw_env.scene.env_origins
    _ar_tm = raw_env.trajectory_manager
    _ar_actterm = (
        raw_env.action_manager._terms.get("right_hand_joint_pos") if hasattr(raw_env.action_manager, "_terms") else None
    )
    _ar_rich = []  # per-frame dicts, saved to the npz

    def _ar_log(frame_idx):
        ofo, fvec = [], []  # per-finger obj-force MAG + VECTOR (object-on-finger, world)
        for s in _ar_objs:
            fm = s.data.force_matrix_w
            if fm is None:
                ofo.append(0.0)
                fvec.append([0.0, 0.0, 0.0])
            else:
                v = fm.sum(dim=(1, 2))[0]  # (3,) object-on-finger
                ofo.append(v.norm().item())
                fvec.append([float(x) for x in v.cpu().numpy()])
        net = _ar_cs.data.net_forces_w
        nfo = [torch.norm(net[:, sid, :], dim=-1)[0].item() if sid >= 0 else 0.0 for sid in _ar_stids]
        tips = _ar_hand.data.body_link_pos_w[:, _ar_tip_ids, :]
        d = _ar_tip_dist(tips, _ar_obj, _ar_surf)[0].detach().cpu().numpy()
        _aud_rows.append((frame_idx, ofo, nfo, [float(x) for x in d]))
        # Rich row: hand-on-object force (Newton's third law: minus the summed object-on-finger forces), object pose
        # and velocity, reference object pose, finger action and residual.
        import numpy as _np3

        fvec = _np3.array(fvec)  # (5,3) object-on-finger
        F_hand_on_obj = -fvec.sum(axis=0)  # (3,) finger->object (sign-flipped), world
        elen = raw_env.episode_length_buf
        try:
            ref = _ar_tm.get_reference_state(elen)
            ref_obj_pos = (ref["right_obj_pos"][0] - _ar_origins[0]).detach().cpu().numpy()
            ref_obj_quat = ref["right_obj_quat"][0].detach().cpu().numpy()
        except (AttributeError, IndexError, KeyError) as _exc:
            note_once("CLIP_AUDIT reference state", _exc)
            ref_obj_pos = _np3.zeros(3)
            ref_obj_quat = _np3.array([1.0, 0.0, 0.0, 0.0])
        row = {
            "frame": frame_idx,
            "F_hand_on_obj": F_hand_on_obj,
            "F_per_finger": (-fvec),
            "obj_pos": (_ar_obj.data.root_pos_w[0] - _ar_origins[0]).detach().cpu().numpy(),
            "obj_quat": _ar_obj.data.root_quat_w[0].detach().cpu().numpy(),
            "obj_lin_vel": _ar_obj.data.root_lin_vel_w[0].detach().cpu().numpy(),
            "ref_obj_pos": ref_obj_pos,
            "ref_obj_quat": ref_obj_quat,
            "eff_frame": int(
                (elen[0].float() / float(getattr(_ar_tm, "_slowdown_factor", 1.0))).floor() + _ar_tm.start_frames[0]
            ),
        }
        if _ar_actterm is not None and hasattr(_ar_actterm, "_combined_actions"):
            # Finger slice of the combined action [6 wrist | n_fingers].
            row["combined_finger"] = _ar_actterm._combined_actions[0, 6:].detach().cpu().numpy()
        try:
            row["residual"] = raw_env.action_manager.action[0].detach().cpu().numpy()
        except (AttributeError, IndexError) as e:  # no action manager / no env 0
            if not _ar_rich:
                print(f"[CLIP_AUDIT] residual action not readable ({e!r}): column left out", flush=True)
        _ar_rich.append(row)

    return _ar_log, _ar_rich


def _audit_report(_aud_rows, _ar_rich, _AR_NAMES, term_b, trunc_b):
    """Print the per-finger contact summary of the finished rollout and save its per-frame arrays."""
    import numpy as _np2

    _objF = _np2.array([r[1] for r in _aud_rows])
    _netF = _np2.array([r[2] for r in _aud_rows])
    _dcm = _np2.array([r[3] for r in _aud_rows]) * 100.0
    _gate = _objF > 0.5
    _st = len(_aud_rows)
    print(
        f"\n[CONTACT-AUDIT] rendered rollout: "
        f"{'FAILURE' if (term_b and not trunc_b) else 'success/timeout'}, "
        f"{_st} frames (== saved video frames). Gate = object-filtered force > 0.5N."
    )
    for _i, _nm in enumerate(_AR_NAMES):
        _fc = _np2.where(_gate[:, _i])[0]
        _rng = f"{int(_fc.min())}-{int(_fc.max())}" if len(_fc) else "-"
        print(
            f"  {_nm:7}: objContact {int(_gate[:, _i].sum()):3d} frames (frames {_rng})  "
            f"maxObjF={_objF[:, _i].max():6.2f}N  maxNetF={_netF[:, _i].max():6.2f}N  "
            f"minDist(body-origin)={_dcm[:, _i].min():4.1f}cm"
        )
    print(f"  robot_contact_frac (NEW objF gate, mean 5x{_st}) = {_gate.mean():.4f}")
    for _r in _aud_rows:
        if max(_r[1]) > 0.5:
            print(
                f"    frame {int(_r[0]):3d} objF: "
                + "  ".join(f"{_AR_NAMES[_i][:3]}={_r[1][_i]:5.1f}" for _i in range(5))
            )
    if _ar_rich:
        _outp = "/tmp/exp_audit_rich.npz"
        _keys = [
            "F_hand_on_obj",
            "F_per_finger",
            "obj_pos",
            "obj_quat",
            "obj_lin_vel",
            "ref_obj_pos",
            "ref_obj_quat",
            "combined_finger",
            "residual",
        ]
        _save = {
            "frame": _np2.array([r["frame"] for r in _ar_rich]),
            "eff_frame": _np2.array([r["eff_frame"] for r in _ar_rich]),
            "objF_mag": _objF,
            "dist_cm": _dcm,
        }
        for _k in _keys:
            if _k in _ar_rich[0]:
                _save[_k] = _np2.array([r[_k] for r in _ar_rich])
        _np2.savez(_outp, **_save)
        print(f"  [audit] rich per-frame arrays saved -> {_outp} ({len(_ar_rich)} frames)")
    sys.stdout.flush()
