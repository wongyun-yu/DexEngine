"""Six-point hand touch measured on the raw capture (five fingertips and a palm point per hand).

The definition (points, near, touched, the measurement margin) belongs to ``dexengine.data.objects.scene_requirement``;
its three thresholds are read through the references that module names (``touch_values``). The builder measures the
same definition on the delivered keypoints against the base tree's object poses (the contact sidecar, stamped as
``case.contact_frac``). Here the raw MANO is posed instead: ``common.measure_objects`` asks ``hand_points`` for the six
points per sample and measures them against the raw object surface; ``common.reduce_window`` asks ``window_touch`` for
the per-object scalars.
"""

from __future__ import annotations

import numpy as np

from dexengine.core import paths as P
from dexengine.data.objects import scene_requirement as SR

_TOUCH: dict = {}
_WRIST_REG: dict = {}


def touch_values() -> dict:
    """{tip_m, palm_m, alpha}, each read from its owner through the reference scene_requirement names (the rule
    registry's reference reader: a measured JSON key or a module constant read from its source). A reference that does
    not resolve to a number is fatal; there is no default."""
    if not _TOUCH:
        from dexengine.data import rules as RR  # noqa: PLC0415

        for k, ref in (("tip_m", SR.TOUCH_TIP_REF), ("palm_m", SR.TOUCH_PALM_REF), ("alpha", SR.TOUCH_ALPHA_REF)):
            val, ok = RR._sym(ref)
            if not ok or not isinstance(val, (int, float)) or isinstance(val, bool):
                raise SystemExit(f"[SCENE_AUDIT] ✗ touch statement: {ref} does not resolve to a number ({val!r})")
            _TOUCH[k] = float(val)
        print(
            f"[TOUCH_DEF] six points · fingertip < {_TOUCH['tip_m'] * 1e3:.1f} mm ({SR.TOUCH_TIP_REF}) · palm < "
            f"{_TOUCH['palm_m'] * 1e3:.1f} mm ({SR.TOUCH_PALM_REF}) · alpha {_TOUCH['alpha']} · oracle margin "
            f"{SR.SLACK_POINT_M * 1e3:.1f} mm",
            flush=True,
        )
    return _TOUCH


def six_points(V: np.ndarray, side: str, tip_verts: list) -> tuple[np.ndarray, np.ndarray]:
    """(fingertips (T,5,3), wrist (T,3)) of posed MANO vertices V (T,778,3) — the points the delivered keypoints
    carry (the fingertip vertices tip_verts · MANO joint 0). The wrist is the model's joint regressor on the posed
    vertices — the dataset modules hand over vertices, not joints; on 400 random poses and shapes per hand it sits
    1.1 mm (median) · 2.5 mm (max) from joint 0, so the palm point moves by 0.6 of that, inside the oracle margin.
    A hand that is not a MANO surface cannot be read this way — its module passes its own points (hand_points)."""
    if V.ndim != 3 or V.shape[1] != 778:
        raise ValueError(f"six_points: {V.shape} is not a MANO surface (T,778,3) — the module must pass its points")
    if side not in _WRIST_REG:
        import smplx  # noqa: PLC0415

        m = smplx.MANO(str(P.mano_model_dir()), is_rhand=(side == "rh"), use_pca=False)
        _WRIST_REG[side] = m.J_regressor[0].double().numpy()
    return V[:, tip_verts], np.einsum("v,tvk->tk", _WRIST_REG[side], V)


def hand_points(hv: dict, six: dict | None, tip_verts: list) -> dict:
    """{"lh"|"rh": (T,6,3)} — the five fingertips then the palm point (SR.palm_point), NaN where the hand is absent.
    six = {"lh"|"rh": (fingertips (T,5,3), wrist (T,3))} from a module whose hand is not a MANO surface; else the
    points are read off hv (six_points)."""
    alpha, out = touch_values()["alpha"], {}
    for s in ("lh", "rh"):
        tips, wrist = six[s] if six is not None else six_points(np.asarray(hv[s], float), s, tip_verts)
        out[s] = np.concatenate([tips, SR.palm_point(wrist, tips, alpha)[:, None]], 1)
    return out


def window_touch(o: dict, idx: np.ndarray, n_win: int) -> dict:
    """The window's touch scalars of one measured object (o["tipd"] (T,5) · o["palmd"] (T,) per hand; idx = its
    window samples; n_win = the window's samples): pt_frac_<hand> = SR.touch_frac of the frames SR.touch_frames
    calls near at the owners' thresholds minus the oracle margin SR.SLACK_POINT_M, and pt_tip_<hand>_mm ·
    pt_palm_<hand>_mm = the window's minimum distances (what the thresholds are compared with)."""
    tv = touch_values()
    tip_m, palm_m = tv["tip_m"] - SR.SLACK_POINT_M, tv["palm_m"] - SR.SLACK_POINT_M
    r = {}
    for s in ("lh", "rh"):
        near = SR.touch_frames(o["tipd"][s][idx], o["palmd"][s][idx], tip_m, palm_m)
        r.update(
            {
                f"pt_frac_{s}": SR.touch_frac(int(near.sum()), n_win),
                f"pt_tip_{s}_mm": round(float(np.min(o["tipd"][s][idx])) * 1e3, 1),
                f"pt_palm_{s}_mm": round(float(np.min(o["palmd"][s][idx])) * 1e3, 1),
            }
        )
    return r


def touch_s(r: dict, n_win: int, dt: float) -> float:
    """The oracle's touch time (s) of one measured object over the window: each hand's near fraction (pt_frac) × the
    window (n_win samples × dt), summed over hands — the quantity the place rule reads off the builder's stamp
    (place_rule.touch_s: case.contact_frac × T / fps), measured here on the raw capture."""
    return round((r["pt_frac_lh"] + r["pt_frac_rh"]) * n_win * dt, 3)


def touched(r: dict, hands=("lh", "rh")) -> int:
    """Any of `hands` touched it: SR.frac_touched of its pt_frac (scene_requirement: any hand)."""
    return int(any(SR.frac_touched(r[f"pt_frac_{s}"]) for s in hands))


def tree_window_end(st: dict, w0: int, w1: int, ti: dict) -> int:
    """w1 clamped to the tree's own length. A dataset module reads a whole clip's window off its name, but a tree is
    shorter than its clip by the tail it drops (segment.frame_map — kin[i] == mano[i], the last frames dropped), and
    with a touch that counts on any frame those samples decide: a hand reaching for the next object inside the dropped
    tail is a touch the builder never saw. Tree length = (segment.end − segment.start) frames at segment.fps, in raw
    frames at the stage's rate (frame step / dt). A tree without that statement keeps the module's window; one that
    maps its frames with an offset is outside the modules' window rule and is rejected."""
    seg = ti.get("segment") or {}
    off = (seg.get("frame_map") or {}).get("offset")
    if off:
        raise ValueError(f"segment.frame_map.offset = {off}: the oracle's window start does not read an offset")
    fr = st["frames"]
    if seg.get("start") is None or seg.get("end") is None or not seg.get("fps") or len(fr) < 2:
        return w1
    raw_hz = float(fr[1] - fr[0]) / st["dt"]
    return min(w1, w0 + int(round((seg["end"] - seg["start"]) / float(seg["fps"]) * raw_hz)))
