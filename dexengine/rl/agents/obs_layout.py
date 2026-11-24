"""Policy-observation layout: term offsets, per-hand token slices, and the checkpoint observation-width check.

The slices follow the obs_v2 term layout exactly (``HandObservationV2`` 39+2F+9T wide, ``ObjectStateV2`` 37+10T
wide, plus an optional object tail); every helper fails loudly when the live layout does not fit. Pure torch, no
Isaac import.
"""

from __future__ import annotations

import torch


def _term_starts(raw_env):
    """(name -> (offset, dim)) for the policy obs group, plus total dim."""
    om = raw_env.unwrapped.observation_manager
    names = om.active_terms["policy"]
    dims = om.group_obs_term_dim["policy"]
    starts = {}
    cursor = 0
    for name, shp in zip(names, dims, strict=True):
        n = int(shp[0]) if isinstance(shp, (tuple, list)) else int(shp)
        starts[name] = (cursor, n)
        cursor += n
    total = om.group_obs_dim["policy"]
    total = int(total[0]) if isinstance(total, (tuple, list)) else int(total)
    return starts, total


def _hand_block_geometry(raw_env, hand_width):
    """(n_finger_obs F, n_tips T) for the live env's HandObservationV2 block.

    Width algebra (dexengine.rl.tasks.mdp.observations.v2.HandObservationV2): W = 39 + 2F + 9T —
    proprio 18+2F (wrist q 9 + dq 6 + gravity 3 + finger q/dq F each),
    wrist tracking 21, tip tracking 9T. T comes from the trajectory manager's
    FK tip list (the same source the obs term reads, same fallback); F is
    solved from the reported width and must come out a positive integer.
    e.g. sharpa_wave F=22 → 128.
    """
    tm = getattr(raw_env.unwrapped, "trajectory_manager", None)
    fk = getattr(tm, "fk_finger_indices", None)
    n_tips = len(fk["tips_list"]) if fk else 5
    rem = hand_width - 39 - 9 * n_tips
    assert rem >= 2 and rem % 2 == 0, f"[OBS-V2] hand block width {hand_width} does not fit 39+2F+9T (T={n_tips})"
    return rem // 2, n_tips


def _structured_obs_slices(raw_env):
    """Purpose slices over the obs_v2 one-hand layout for --policy_arch structured.

    hand(39+2F+9T) = [proprio [0, 18+2F) | tracking rest]; last_base_action+actions
    → action-context (joined with proprio); object_state(37+10T) → object branch.
    Fails loud unless the layout is EXACTLY the obs_v2 one-hand set — no silent fallbacks.
    """
    om = raw_env.unwrapped.observation_manager
    names = om.active_terms["policy"]
    dims = om.group_obs_term_dim["policy"]
    pa, tr, obj = [], [], []
    n_tips = None
    cursor = 0
    for name, shp in zip(names, dims, strict=True):
        n = int(shp[0]) if isinstance(shp, (tuple, list)) else int(shp)
        rng = list(range(cursor, cursor + n))
        if name in ("right_hand", "left_hand"):  # a left-hand build names its hand term 'left_hand'
            n_fing, n_tips = _hand_block_geometry(raw_env, n)
            pp = 18 + 2 * n_fing
            pa += rng[:pp]  # q(9+F) + dq(6+F) + gravity(3)
            tr += rng[pp:]  # wrist tracking(21) + tip tracking(9T)
        elif name in ("last_base_action", "actions"):
            pa += rng
        elif name == "object_state":
            assert n_tips is not None, "[STRUCTURED] object_state precedes hand term"
            assert n == 37 + 10 * n_tips, f"[STRUCTURED] object_state {n} != 37+10T (T={n_tips})"
            obj += rng
        else:
            raise ValueError(f"[STRUCTURED] unexpected obs term '{name}' — uni obs_v2 layout only")
        cursor += n
    total = om.group_obs_dim["policy"]
    total = int(total[0]) if isinstance(total, (tuple, list)) else int(total)
    assert cursor == total and len(pa) + len(tr) + len(obj) == total, "[STRUCTURED] slice coverage broken"
    print(f"  [STRUCTURED] slices: proprio+action={len(pa)} tracking={len(tr)} object={len(obj)} (total {total})")
    return {"proprio_action": pa, "tracking": tr, "object": obj}


#: Articulated object: ObjectStateV2 grows by 2 trailing dims (current joint angle, reference minus angle). The token
#: layer appends them to the obj_ref token (reference tracking 18 -> 20), so the token count (7 + 2T) is unchanged and a
#: rigid run has no extra dims.
ARTIC_OBJ_EXTRA = 2
#: Fixed object layout (--obj_obs_fixed 1): every object carries (θ, ref−θ, valid bit) — rigid 0,0,0 — so the tail is 3
#: for rigid AND articulated clips; the three dims join the obj_ref token the same way (token count unchanged).
#: These two counts are an oracle over the width the obs term (dexengine.rl.tasks.mdp.observations.v2
#: ARTIC/FIXED_TAIL_DIMS) reports.
FIXED_OBJ_EXTRA = 3


def _object_extra_dims(width: int, n_tips: int, label: str) -> int:
    """Trailing object dims implied by an ObjectStateV2 width: 0 (rigid), ARTIC_OBJ_EXTRA (articulated) or
    FIXED_OBJ_EXTRA (fixed layout); any other width fails the assertion."""
    extra = int(width) - (37 + 10 * int(n_tips))
    assert extra in (0, ARTIC_OBJ_EXTRA, FIXED_OBJ_EXTRA), (
        f"[TOKEN] {label} {width} != 37+10T(+{ARTIC_OBJ_EXTRA} articulated | +{FIXED_OBJ_EXTRA} A-5 fixed) (T={n_tips})"
    )
    if extra == ARTIC_OBJ_EXTRA:
        print(
            f"  [TOKEN][ARTIC] {label}: articulated observation +{extra} (θ, ref−θ) → folded into the obj_ref token "
            "(token count unchanged)",
            flush=True,
        )
    elif extra == FIXED_OBJ_EXTRA:
        print(
            f"  [TOKEN][OBJ_OBS_FIXED] {label}: A-5 fixed tail +{extra} (θ, ref−θ, valid bit) → folded into the "
            "obj_ref token (token count unchanged)",
            flush=True,
        )
    return extra


def checkpoint_obs_dim(ckpt: dict) -> int | None:
    """The policy-input width a saved policy was trained on — the state preprocessor's running mean (skrl), else
    1 + the largest token index a token trunk carries. None when the checkpoint states neither."""
    sp = ckpt.get("state_preprocessor") if isinstance(ckpt, dict) else None
    rm = (sp or {}).get("running_mean") if isinstance(sp, dict) else None
    if rm is not None and hasattr(rm, "numel"):
        return int(rm.numel())
    pol = ckpt.get("policy") if isinstance(ckpt, dict) else None
    idx = [int(v.max()) for k, v in (pol or {}).items() if "._idx_" in k and hasattr(v, "max") and v.numel()]
    if idx:
        return 1 + max(idx)
    # an MLP trunk without a preprocessor: the first layer's input width (the keys the bimanual RESUME DIM-GUARD reads)
    w = next((pol[k] for k in ("encoder.0.weight", "net_container.0.weight") if k in (pol or {})), None)
    return int(w.shape[1]) if w is not None and getattr(w, "ndim", 0) == 2 else None


def assert_checkpoint_obs_dim(path: str, live_dim: int, where: str) -> None:
    """Die before loading a checkpoint whose observation width differs from this env's. The load itself would fail
    on a shape mismatch deep inside skrl; this names the likely cause — the observation layout is a checkpoint
    contract (--obj_obs_fixed, the hand's width, an articulated object) and must match the training run.
    Every load site calls this right before load(), so it is also the one place the same dict is checked for
    non-finite weights/statistics ([CKPT_FINITE], owner nonfinite_guard — a diverged checkpoint never loads)."""
    from dexengine.rl.train.nonfinite_guard import assert_checkpoint_finite

    ck = torch.load(path, map_location="cpu", weights_only=False)
    got = checkpoint_obs_dim(ck)
    if got is None:
        raise RuntimeError(
            f"[CKPT-OBS] {where}: the checkpoint does not state its observation width "
            f"(no preprocessor, token index or first-layer weight) — not loaded unchecked: {path}"
        )
    if got != int(live_dim):
        raise RuntimeError(
            f"[CKPT-OBS] {where}: checkpoint observation width {got} ≠ current env {live_dim} — "
            f"{obs_width_remedy(int(live_dim) - got)}: {path}"
        )
    print(f"  [CKPT-OBS] {where}: observation width {got} = env {live_dim} ✓", flush=True)
    assert_checkpoint_finite(ck, where, path)


def obs_width_remedy(d: int) -> str:
    """The remedy a width gap (live env − checkpoint) names. The object-layout switch moves each side's object tail by
    FIXED_OBJ_EXTRA (rigid, 0 -> 3) or FIXED_OBJ_EXTRA - ARTIC_OBJ_EXTRA (articulated, 2 -> 3), so a layout gap is 1..6
    on a two-sided policy and its sign says which layout the checkpoint has; any other gap is not the layout."""
    per_side = (FIXED_OBJ_EXTRA - ARTIC_OBJ_EXTRA, FIXED_OBJ_EXTRA)
    if d == 0 or abs(d) > 2 * max(per_side):
        return (
            f"width gap {d:+d} is not explained by the object observation layout ({per_side[0]}·{per_side[1]} "
            f"columns per side) — another observation contract (hand, articulation, …) differs from training"
        )
    want = 0 if d > 0 else 1
    return (
        f"width gap {d:+d} looks like an object observation layout difference: the checkpoint has layout "
        f"{want} → set OBJ_OBS_FIXED={want} in the launch env. The run's value is the `OBJ_OBS_FIXED=` token "
        f"of the training manifest <RUN>_cmd.txt, not its date (no or empty token = layout 0 · without the "
        f"manifest on this machine do not guess 0, read the training machine's) · value owner = the launch "
        f"table scripts/_rl_recipe.sh"
    )


def _token_dict(rh0, lba_idx, act_idx, ob0, n_fing, n_tips, n_obj_extra=0):
    """Build the semantic-token index lists for ONE hand, given its block offsets.

    rh0 = offset of the (39+2F+9T)-dim HandObservationV2 block; lba_idx/act_idx =
    this hand's last_base_action and action absolute indices; ob0 = offset of the
    (37+10T)-dim ObjectStateV2 block. Derived from the exact dexengine.rl.tasks.mdp.observations.v2 layout.
    Token count = 7 + 2T (17 for T=5).
    n_obj_extra: trailing object dims after ObjectStateV2 (0, ARTIC_OBJ_EXTRA or FIXED_OBJ_EXTRA), appended to obj_ref.
    """
    F, T = n_fing, n_tips
    toks = {}
    toks["q"] = list(range(rh0, rh0 + 9 + F))
    toks["dq"] = list(range(rh0 + 9 + F, rh0 + 15 + 2 * F))
    toks["ctx"] = list(range(rh0 + 15 + 2 * F, rh0 + 18 + 2 * F)) + list(lba_idx) + list(act_idx)
    w0 = rh0 + 18 + 2 * F
    toks["wrist_track"] = list(range(w0, w0 + 21))
    t0 = w0 + 21
    for i in range(T):  # tip tracking: delta_pos(3T)+ref_vel(3T)+delta_vel(3T), per-tip gather
        toks[f"tip_track_{i}"] = (
            list(range(t0 + i * 3, t0 + i * 3 + 3))
            + list(range(t0 + 3 * T + i * 3, t0 + 3 * T + i * 3 + 3))
            + list(range(t0 + 6 * T + i * 3, t0 + 6 * T + i * 3 + 3))
        )
    toks["obj_kin"] = list(range(ob0, ob0 + 15))
    toks["obj_ref"] = list(range(ob0 + 15, ob0 + 33))
    c0 = ob0 + 33
    for i in range(T):  # per-tip contact: surface_vec(3)+normal(3)+signed_dist(1)+force(3)
        toks[f"tip_contact_{i}"] = (
            list(range(c0 + i * 3, c0 + i * 3 + 3))
            + list(range(c0 + 3 * T + i * 3, c0 + 3 * T + i * 3 + 3))
            + [c0 + 6 * T + i]
            + list(range(c0 + 7 * T + i * 3, c0 + 7 * T + i * 3 + 3))
        )
    toks["phys"] = list(range(c0 + 10 * T, c0 + 10 * T + 4))
    if n_obj_extra:
        e0 = c0 + 10 * T + 4
        toks["obj_ref"] = toks["obj_ref"] + list(range(e0, e0 + int(n_obj_extra)))
    return toks


def _token_obs_slices(raw_env):
    """Tokenize the obs_v2 one-hand layout into 17 semantic tokens for --policy_arch token.

    The hand term is 'right_hand' or, for a left-hand build, 'left_hand' — token offsets do not depend on the name,
    only the lookup key differs.
    """
    starts, total = _term_starts(raw_env)
    hand_key = "right_hand" if "right_hand" in starts else "left_hand"
    assert hand_key in starts, "[TOKEN] requires obs_v2 uni (hand term missing)"
    n_fing, n_tips = _hand_block_geometry(raw_env, starts[hand_key][1])
    ow = starts.get("object_state", (0, 0))[1]
    _oe = _object_extra_dims(ow, n_tips, "object_state")
    lba0, lba_n = starts["last_base_action"]
    act0, act_n = starts["actions"]
    toks = _token_dict(
        starts[hand_key][0],
        range(lba0, lba0 + lba_n),
        range(act0, act0 + act_n),
        starts["object_state"][0],
        n_fing=n_fing,
        n_tips=n_tips,
        n_obj_extra=_oe,
    )
    covered = sorted(sum(toks.values(), []))
    assert covered == list(range(total)), "[TOKEN] token slices do not exactly cover the obs"
    print(
        f"  [TOKEN] {len(toks)} tokens over {total} dims (hand={hand_key}): "
        + ", ".join(f"{k}({len(v)})" for k, v in toks.items())
    )
    return toks


def _token_slices_bimanual(raw_env):
    """Per-hand token slices for the obs_v2 BIMANUAL layout (two-tower token policies).

    Returns (right_dict, left_dict, combined_dict). Right/left each = 7+2T tokens over
    that hand's slice (hand 39+2F+9T + last_base_action + its half of actions + object
    37+10T). 'actions' (2A) is split [0:A]→right, [A:2A]→left. combined
    (left keys suffixed '__L') is for the centralized critic. Fails loud on mismatch.
    """
    starts, total = _term_starts(raw_env)
    for k in ("right_hand", "object_state", "left_hand", "object_state_left"):
        assert k in starts, f"[TOKEN-BI] requires obs_v2 bimanual ({k} missing)"
    fr, tr_ = _hand_block_geometry(raw_env, starts["right_hand"][1])
    fl, tl_ = _hand_block_geometry(raw_env, starts["left_hand"][1])
    _oe_r = _object_extra_dims(starts["object_state"][1], tr_, "[TOKEN-BI] object_state")
    _oe_l = _object_extra_dims(starts["object_state_left"][1], tl_, "[TOKEN-BI] object_state_left")
    lba0, lban = starts["last_base_action"]
    act0, actn = starts["actions"]
    assert actn % 2 == 0, f"[TOKEN-BI] actions dim {actn} odd — cannot split R/L"
    half = actn // 2
    llba0, llban = starts["left_last_base_action"]
    right = _token_dict(
        starts["right_hand"][0],
        range(lba0, lba0 + lban),
        range(act0, act0 + half),
        starts["object_state"][0],
        n_fing=fr,
        n_tips=tr_,
        n_obj_extra=_oe_r,
    )
    left = _token_dict(
        starts["left_hand"][0],
        range(llba0, llba0 + llban),
        range(act0 + half, act0 + actn),
        starts["object_state_left"][0],
        n_fing=fl,
        n_tips=tl_,
        n_obj_extra=_oe_l,
    )
    # Arm-joint residual (DEXENGINE_ARM_JOINT_RESIDUAL): each hand's arm-joint obs term, appended last, is one more
    # token of that hand. Absent when the switch is off (token set unchanged).
    for _k, _d in (("arm_joint", right), ("arm_joint_left", left)):
        if _k in starts:
            _a0, _an = starts[_k]
            _d["arm_joint"] = list(range(_a0, _a0 + _an))
            print(f"  [TOKEN-BI] {_k}: arm-joint token {_an} dims", flush=True)
    combined = dict(right)
    for k, v in left.items():
        combined[f"{k}__L"] = v
    cov = sorted(sum(combined.values(), []))
    assert cov == list(range(total)), "[TOKEN-BI] per-hand token slices do not exactly cover the obs"
    print(f"  [TOKEN-BI] right={len(right)} left={len(left)} tokens over {total} dims")
    return right, left, combined
