"""Kinematic pair trees (prep_tree.kin_ik): the per-frame IK steps — FK error and reach measure, our-arm seed
candidates, DLS from the previous solution with multi-start restart and the divergence guard."""

from __future__ import annotations


def fk_err(model, data, ee_id, qq, tp, tr):
    """FK error (Euclidean m, geodesic °) + reach measure max(position/POS_TOL, rotation/ROT_TOL) — ≤ 1 means the
    frame reaches.
    """
    import numpy as np  # noqa: PLC0415
    import pinocchio as pin  # noqa: PLC0415

    from experiments.policies.kin_data import reach as KR  # noqa: PLC0415 — owner of the reach measure (tolerances)

    pin.forwardKinematics(model, data, qq)
    pin.updateFramePlacements(model, data)
    mf = data.oMf[ee_id]
    dp = float(np.linalg.norm(mf.translation - tp))
    dr = float(np.degrees(np.linalg.norm(pin.log3(mf.rotation.T @ tr))))
    return max(dp / KR.POS_TOL_M, dr / KR.ROT_TOL_DEG), dp, dr


def _qfull(model, qidx, v):
    import pinocchio as pin  # noqa: PLC0415

    q0 = pin.neutral(model)
    for k_, qi in enumerate(qidx):
        q0[qi] = v[k_]
    return q0


def seed_step(model, data, ee_id, qidx, t, tp, tr, q, S_q, S_al, s_pe, s_re):
    """Frame t with our arm seed: candidates = DLS (our arm), our arm as is, DLS (previous), previous — among those
    that reach, the one closest to our arm (else to the previous); none reaches → smallest reach measure. Our arm's FK
    error goes to s_pe / s_re. → (solution, its source)."""
    import numpy as np  # noqa: PLC0415

    from dexengine.placement import arm_ik as AIK  # noqa: PLC0415

    st = S_q[0] if t == 0 else (S_q[t] if S_al else None)
    cands, ref = [], None
    if st is not None:
        qs = _qfull(model, qidx, st)
        _sc, s_pe[t], s_re[t] = fk_err(model, data, ee_id, qs, tp, tr)
        qd = AIK.solve_ik_pose(model, data, tp, tr, ee_id, qidx, max_iters=300, q0=qs)[0]
        cands += [(qd, 5), (qs, 6)]
        ref = np.asarray(st, float)
    if q is not None:
        cands += [(AIK.solve_ik_pose(model, data, tp, tr, ee_id, qidx, max_iters=300, q0=q)[0], 7), (q, 8)]
        ref = np.array([q[i] for i in qidx]) if ref is None else ref
    sc = [fk_err(model, data, ee_id, c, tp, tr)[0] for c, _ in cands]
    okc = [k_ for k_, v_ in enumerate(sc) if v_ <= 1.0]
    k = (
        min(okc, key=lambda k_: float(np.abs(np.array([cands[k_][0][i] for i in qidx]) - ref).max()))
        if okc
        else int(np.argmin(sc))
    )
    return cands[k][0], cands[k][1]


def dls_step(model, data, ee_id, qidx, t, tp, tr, q, a_tol_m, a_tol_deg, restarts, retry_every, prev_ok, last_re):
    """A later frame without a seed: DLS from the previous solution, a multi-start restart when it does not reach,
    then the divergence guard. → (solution, source or None (= 0), last restart frame, restarted, guard taken)."""
    import numpy as np  # noqa: PLC0415
    import pinocchio as pin  # noqa: PLC0415

    from dexengine.placement import arm_ik as AIK  # noqa: PLC0415

    src_t, restarted, guarded = None, False, False
    q_prev = q
    q, e_p, e_r, _ = AIK.solve_ik_pose(model, data, tp, tr, ee_id, qidx, max_iters=300, q0=q)
    cands = [q, q_prev]
    bad = e_p > a_tol_m or e_r > a_tol_deg
    if bad and (
        prev_ok or t - last_re >= retry_every
    ):  # previous solution does not reach → multi-start (quality, not rejection) ·
        last_re = t  # a run that keeps failing restarts only every retry_every frames
        # continuity: only candidates that reach (within tolerance) are taken, the one closest to the previous
        # solution — no switch to a candidate that does not reach
        # (jumping a 2π branch for a non-reaching improvement flips the arm in replay and inflates joint-speed
        # defects).
        rng = np.random.default_rng(t)
        qp = np.array([q[i] for i in qidx])
        best = None
        for _s in range(restarts):
            q0 = pin.neutral(model)
            for qi in qidx:
                lo, hi = model.lowerPositionLimit[qi], model.upperPositionLimit[qi]
                q0[qi] = rng.uniform(lo, hi) if np.isfinite(lo) and np.isfinite(hi) else rng.uniform(-1.5, 1.5)
            q2, p2, r2, _ = AIK.solve_ik_pose(model, data, tp, tr, ee_id, qidx, max_iters=300, q0=q0)
            if p2 > a_tol_m or r2 > a_tol_deg:
                continue
            dist = float(np.abs(np.array([q2[i] for i in qidx]) - qp).max())
            if best is None or dist < best[0]:
                best = (dist, q2, p2, r2)
        if best is not None:
            q, e_p, e_r = best[1], best[2], best[3]
            restarted = True
            src_t = 1
    # divergence guard (continued DLS divergence once carried a solution 0.77 m off for 250 frames while
    # multi-start's best
    # was 0.07–0.12 m): only when the rule above leaves the reach tolerance, take the better (smaller reach
    # measure) of the
    # previous solution's DLS and the previous solution itself (never worse than previous). Restart candidates
    # are not added —
    # they jumped to another branch on unreachable frames and flipped the arm. Frames within tolerance follow
    # the rule above.
    if fk_err(model, data, ee_id, q, tp, tr)[0] > 1.0:
        sc = [fk_err(model, data, ee_id, c, tp, tr)[0] for c in cands]
        k = int(np.argmin(sc))
        if sc[k] < fk_err(model, data, ee_id, q, tp, tr)[0]:
            q = cands[k]
            guarded = True
            src_t = 2 + k
    return q, src_t, last_re, restarted, guarded
