"""Report half of the reference-base audit (refbase_audit): per-band tracking errors, both-hands wrist tracking,
the object-mass cache check and the fingertip divergence, printed from the arrays the rollout filled."""

from __future__ import annotations

from dexengine.rl.tasks import env_flags


def report_tracking_bands(
    raw_env,
    T,
    _bimanual,
    _np,
    arm_kd_used,
    arm_kp_used,
    fnames,
    os,
    slowdown,
    tip_names,
    F,
    fte,
    q_ref_arr,
    q_robot_arr,
    qe,
    we,
    we_l,
    wre,
    wre_l,
):
    """Print the per-band tracking errors of the rollout, then the both-hands report."""

    # ---- report ----
    def band(a, lo, hi):  # mean over [lo,hi) ignoring NaN; a is (T,) or (T,k)
        seg = a[lo : min(hi, T)]
        seg = seg[~_np.isnan(seg).any(axis=-1)] if seg.ndim == 2 else seg[~_np.isnan(seg)]
        if seg.size == 0:
            return _np.nan
        return float(_np.nanmean(seg))

    bands = [
        ("approach 0-27", 0, 27),
        ("crossing 27-38", 27, 38),
        ("near-grasp/lift 38-70", 38, 70),
        ("post-grasp 70-140", 70, 140),
        ("scrape 140-end", 140, T),
    ]
    print("\n  ---- per-band tracking error (PD reference base, residual=0) ----")
    print(f"  {'band':24} {'wristPos_cm':>11} {'wristRot_deg':>12} {'fingerQpos_rad':>15} {'fingertip_cm':>12}")
    for name, lo, hi in bands:
        wp_cm = band(we, lo, hi) * 100
        wr_d = band(wre, lo, hi)
        fq = band(qe, lo, hi)
        ftc = band(fte, lo, hi) * 100
        print(f"  {name:24} {wp_cm:>11.3f} {wr_d:>12.3f} {fq:>15.4f} {ftc:>12.3f}")
    # overall
    print(
        f"  {'OVERALL':24} {_np.nanmean(we) * 100:>11.3f} {_np.nanmean(wre):>12.3f} "
        f"{_np.nanmean(qe):>15.4f} {_np.nanmean(fte) * 100:>12.3f}"
    )

    _report_both_hands_wrist_tracking(
        raw_env,
        F,
        T,
        _bimanual,
        _np,
        arm_kd_used,
        arm_kp_used,
        band,
        bands,
        fnames,
        fte,
        os,
        q_ref_arr,
        q_robot_arr,
        qe,
        slowdown,
        tip_names,
        we,
        we_l,
        wre,
        wre_l,
    )


def _report_both_hands_wrist_tracking(
    raw_env,
    F,
    T,
    _bimanual,
    _np,
    arm_kd_used,
    arm_kp_used,
    band,
    bands,
    fnames,
    fte,
    os,
    q_ref_arr,
    q_robot_arr,
    qe,
    slowdown,
    tip_names,
    we,
    we_l,
    wre,
    wre_l,
):
    """BOTH-HANDS wrist tracking (the bimanual wrist-fix verification)."""
    _report_both_hands_wrist(
        raw_env,
        F,
        T,
        _bimanual,
        _np,
        arm_kd_used,
        arm_kp_used,
        band,
        bands,
        fnames,
        fte,
        os,
        q_ref_arr,
        q_robot_arr,
        qe,
        slowdown,
        tip_names,
        we,
        we_l,
        wre,
        wre_l,
    )


def _report_both_hands_wrist(
    raw_env,
    F,
    T,
    _bimanual,
    _np,
    arm_kd_used,
    arm_kp_used,
    band,
    bands,
    fnames,
    fte,
    os,
    q_ref_arr,
    q_robot_arr,
    qe,
    slowdown,
    tip_names,
    we,
    we_l,
    wre,
    wre_l,
):
    """Both-hands wrist tracking (the bimanual wrist-fix verification)."""
    arm_ff, grav_off, save = _report_wrist_and_divergence(
        raw_env,
        F,
        _bimanual,
        _np,
        arm_kd_used,
        arm_kp_used,
        band,
        bands,
        fnames,
        fte,
        os,
        qe,
        tip_names,
        we,
        we_l,
        wre,
        wre_l,
    )
    _np.savez(
        save,
        wrist_pos_err=we,
        wrist_rot_err_deg=wre,
        finger_qpos_err=qe,
        fingertip_err=fte,
        obj_force=F,
        q_robot=q_robot_arr,
        q_ref=q_ref_arr,
        wrist_pos_err_left=we_l,
        wrist_rot_err_deg_left=wre_l,
        bimanual=_bimanual,
        finger_names=_np.array(fnames),
        tip_names=_np.array(tip_names),
        slowdown=slowdown,
        T=T,
        arm_kp=arm_kp_used,
        arm_kd=arm_kd_used,
        grav_off=grav_off,
        arm_ff=arm_ff,
    )
    print(f"\n  saved per-frame audit arrays -> {save}")
    print("=" * 78 + "\n")


def _report_wrist_and_divergence(
    raw_env,
    F,
    _bimanual,
    _np,
    arm_kd_used,
    arm_kp_used,
    band,
    bands,
    fnames,
    fte,
    os,
    qe,
    tip_names,
    we,
    we_l,
    wre,
    wre_l,
):
    """Both-hands wrist tracking (bimanual), then the fingertip divergence report."""
    # ---- BOTH-HANDS wrist tracking (the bimanual wrist-fix verification) ----
    if _bimanual:
        print("\n  ---- BOTH-HANDS wrist target-vs-current (PD reference base, residual=0) ----")
        print(f"  {'band':24} {'R_wPos_cm':>10} {'R_wRot_deg':>11} {'L_wPos_cm':>10} {'L_wRot_deg':>11}")
        for name, lo, hi in bands:
            print(
                f"  {name:24} {band(we, lo, hi) * 100:>10.3f} {band(wre, lo, hi):>11.3f} "
                f"{band(we_l, lo, hi) * 100:>10.3f} {band(wre_l, lo, hi):>11.3f}"
            )
        print(
            f"  {'OVERALL':24} {_np.nanmean(we) * 100:>10.3f} {_np.nanmean(wre):>11.3f} "
            f"{_np.nanmean(we_l) * 100:>10.3f} {_np.nanmean(wre_l):>11.3f}"
        )
        print(
            "  (LEFT rot err near 180 deg => the Rz(+90) mount fix did NOT take; "
            "near right's value => left wrist now tracks correctly.)"
        )

        _report_object_mass_cache(raw_env)

    arm_ff, grav_off, save = _report_fingertip_divergence(
        F, _np, arm_kd_used, arm_kp_used, fnames, fte, os, qe, tip_names
    )
    return arm_ff, grav_off, save


def _report_object_mass_cache(raw_env):
    """Object-mass cache check (per-object keying of the right and left masses)."""
    # ---- object-mass cache verification (per-object keying, task-1 fix) ----
    mr = getattr(raw_env, "_object_mass_total_object", None)
    ml = getattr(raw_env, "_object_mass_total_object_left", None)
    print("\n  ---- object-mass cache (per-object keying check) ----")
    print(
        f"    right _object_mass_total_object      = {float(mr.mean().item()):.5f} kg"
        if mr is not None
        else "    right cache: NOT POPULATED"
    )
    print(
        f"    left  _object_mass_total_object_left = {float(ml.mean().item()):.5f} kg"
        if ml is not None
        else "    left  cache: NOT POPULATED"
    )
    if mr is not None and ml is not None:
        # Per-object keying is proven by the two SEPARATE cache attrs existing and being
        # read from each object's own physx view — NOT by the values differing. Under the
        # Task-5 single-source policy both objects are 0.2 kg, so equal values are EXPECTED
        # (not contamination). The pre-fix bug would instead show object_left silently
        # holding the RIGHT object's value when the right was overridden to a different mass.
        print(
            f"    per-object cache attrs present (keying OK); "
            f"left/right ratio = {float(ml.mean().item() / max(mr.mean().item(), 1e-9)):.2f} "
            f"(== 1.00 expected under the 0.2 kg single-source default)"
        )


def _report_fingertip_divergence(F, _np, arm_kd_used, arm_kp_used, fnames, fte, os, qe, tip_names):
    """Fingertip-tracking divergence report."""
    # fingertip-tracking divergence (the termination that fires first). Report first frame each
    # tip exceeds a few thresholds, and the worst-tip max.
    print("\n  ---- fingertip-tracking divergence (env-local, cm) ----")
    ft_max_per_tip = _np.nanmax(fte, axis=0)
    print("    max fingertip err (cm): " + "  ".join(f"{tip_names[i]}={ft_max_per_tip[i] * 100:.2f}" for i in range(5)))
    worst = _np.nanmax(fte, axis=1)  # (T,) worst tip each frame
    for TH in (0.03, 0.05, 0.08):
        idx = _np.where(worst > TH)[0]
        first = int(idx[0]) if idx.size else None
        print(f"    first frame worst-tip > {TH * 100:.0f} cm: {first}")

    # contact/force timing
    print("\n  ---- object-filtered contact/force timing ----")
    for TH in (0.5,):
        act = F > TH

        def first(mask):
            idx = _np.where(mask)[0]
            return int(idx[0]) if idx.size else None

        per = {tip_names[i]: first(act[:, i]) for i in range(5)}
        ti_both = first(act[:, 0] & act[:, 1])
        print(f"    threshold {TH} N — first contact: " + "  ".join(f"{n}={per[n]}" for n in tip_names))
        print(
            f"      first thumb+index BOTH: {ti_both}  "
            f"maxF(N): " + "  ".join(f"{tip_names[i]}={_np.nanmax(F[:, i]):.1f}" for i in range(5))
        )

    # hand-shape sanity: per-frame total finger-qpos deviation; flag frames where any joint
    # deviates a lot from the reference (sign the PD base cannot hold the commanded shape).
    print("\n  ---- hand-shape (finger qpos) sanity ----")
    qmax_per_joint = _np.nanmax(qe, axis=0)
    print(
        "    max |qpos err| per joint (rad): "
        + "  ".join(
            f"{fnames[i].split('_joint')[0] if '_joint' in fnames[i] else fnames[i]}={qmax_per_joint[i]:.3f}"
            for i in range(6)
        )
    )
    print(f"    mean |qpos err| over clip (rad): {_np.nanmean(qe):.4f}")

    # Tag the output by effective gains + gravity + feedforward so a sweep doesn't overwrite.
    grav_off = env_flags.get("DEXENGINE_DISABLE_ROBOT_GRAVITY") == "1"
    # Default 1 (on), matching the residual action's default. The value only tags the diagnostic npz file name
    # (no physics effect); set DEXENGINE_ARM_FF=0 to reproduce the old default.
    arm_ff = env_flags.get("DEXENGINE_ARM_FF") == "1"
    _tag = ""
    if env_flags.get("DEXENGINE_ARM_KP") or env_flags.get("DEXENGINE_ARM_KD") or grav_off or arm_ff:
        _tag = f"_kp{arm_kp_used:.0f}_kd{arm_kd_used:.0f}_g{'off' if grav_off else 'on'}_ff{'on' if arm_ff else 'off'}"
    from dexengine.core.paths import run_out  # under the run tree, not the code tree or the cwd

    save = str(run_out(f"results/taco/refbase_pd_audit_f1_knife{_tag}.npz"))
    os.makedirs(os.path.dirname(save), exist_ok=True)
    return arm_ff, grav_off, save
