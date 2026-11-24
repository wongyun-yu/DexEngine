"""f0 object repair of one side over the spawn-hold window, the solve (rule: repair_obj).

ONE solver on ONE measure: a damped Gauss-Newton projection of the hand pose on the main object's cooked hull (the
shape PhysX collides). The unknown is dq = (wrist translation 3 · wrist rotation 3 · independent fingers), a change
of the frame's own data pose. The residual rows are
  overlap   each (hand collision piece, object piece) pair: SAT depth (face axes of both, as repair_obj.sat_depth)
            above the target of the piece's body (its class's residual, repair_obj.THRESHOLDS_MM); the row moves the
            deepest point of the pair along the separating axis, so a link that is inside ends on the surface, not
            past it
  contact   each link touching the object at this frame (repair_obj.OBJ_TOUCH_MM on the visual mesh): the distance of
            its nearest vertex to the main object's cooked hull above OBJ_TOUCH_MM
  still     each pinned still on triangles (repair_still.rows): per hand piece, its deepest hull vertex inside the still
            by more than the still's target (the solve grid's value; it leaves along the grid gradient) and the
            deepest still surface sample inside the piece (it leaves along the piece's face normal)
  desk      each link's lowest vertex below the desk floor (the frame's own deepest desk vertex: no deeper; frame 0 of
            a desk-deep start: the desk plane + the desk repair's clearance, repair_obj.desk_start)
  movement  LAMBDA · |w_dq · dq|² (the repair's cost weights) — breaks ties only
Point Jacobians are finite differences of the link transforms (nd + 1 forward-kinematics calls per iteration).

THE WINDOW. Frame 0 is solved from the data pose; every later window frame is solved warm-started from the previous
frame's change (one smooth path through the window, not independent per-frame solutions). A frame passes when its
SAT depth is ≤ OBJ_RESIDUAL_MM, its desk is no deeper and every touching link is still within CONTACT_BAND of the
cooked hull. At frame 0, when no attempt passes, the least-SAT attempt that passes the same checks at the open
thresholds (repair_obj.failed_checks k=0) is kept and marked (`obj_kept`); a frame-0 side with no such attempt is
refused (SKIP_OBJ; the prejudge holds the clip); a later failure keeps the previous frame's change for that frame
(recorded per frame in the report, `obj_window_carried`).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import ConvexHull

from dexengine.retarget.f0 import repair_obj as RO
from dexengine.retarget.f0 import repair_still as RST
from dexengine.retarget.f0.repair_cfg import CONTACT_BAND, HC, STRICT_MARGIN_SCALE
from dexengine.retarget.f0.repair_geom import _aa_to_R
from dexengine.retarget.f0.repair_solve import _close_report, _repaired_fields

#: weight of the movement term against the squared residual rows (m²): a tie-breaker only
LAMBDA = 1e-6
#: Gauss-Newton iterations per solve
ITERS = 60
#: finite-difference step of the point Jacobians
EPS = 1e-6


#: _touch_min: vertices kept by the carried bound above this share of a link are all evaluated (the bound saves
#: nothing there)
SDF_BOUND_FULL = 0.5
#: _overlap_rows: a pair whose last separating axis still gives a depth this far below the target is skipped (m; covers
#: the rounding of one plane's values against the full product, ~1e-16 m at metre scale)
SEP_EPS_M = 1e-12


def _hull(v):
    """(hull vertices, plane equations) of a convex piece; (None, None) for a flat piece."""
    try:
        h = ConvexHull(v)
        return v[h.vertices], h.equations
    except Exception:  # noqa: BLE001 — qhull refuses flat / degenerate sets: no volume, no overlap
        return None, None


class Projection:
    """The projection problem of one side at one frame. pose = (wrist position, wrist R, independent fingers);
    desk_floor = the lowest z a hand vertex may reach; box = (lo, hi) of the independent fingers; touch = names of the
    links touching the object at this frame."""

    def __init__(self, strict, cobj, oR, opos, pose, desk_floor, box, touch):
        self.s = strict
        self.wpos0, self.wR0, self.indep0 = pose
        self.floor = desk_floor
        self.lo, self.hi = box
        self.names = [ln for ln, _fid, _p, _dl in strict.links]
        self.fids = [fid for _ln, fid, _p, _dl in strict.links]
        self.pieces = []  # (link index, hull vertices (link frame), planes (link frame))
        for li, ln in enumerate(self.names):
            for p in strict.parts.get(ln, ()):
                v, e = _hull(np.asarray(p, float))
                if v is not None:
                    self.pieces.append((li, v, e))
        self.obj = []  # (world vertices, world planes, box lo, box hi)
        for v, e, _lo, _hi in cobj.pieces:
            vw = v @ oR.T + opos
            n = e[:, :3] @ oR.T
            self.obj.append((vw, np.c_[n, e[:, 3] - n @ opos], vw.min(0), vw.max(0)))
        body = [0] * len(self.obj) if cobj.body_of is None else list(cobj.body_of)
        #: the pinned stills on triangles with this side's solve grid (repair_still), after the object pieces
        self.stills = [(b, b.grids[strict.side]) for b in getattr(cobj, "stills", ())]
        self.gates = [RST.gate_planes(e) for _li, _v, e in self.pieces] if self.stills else []  # RST.rows' gate
        #: each object piece's residual (mm): its body class's (repair_obj.THRESHOLDS_MM); then one per still
        self.res_mm = np.array(
            [RO.THRESHOLDS_MM[cobj.bodies[b][1]][1] for b in body] + [RO.THRESHOLDS_MM["still"][1]] * len(self.stills),
            float,
        )
        main = [o for o, b in zip(self.obj, body, strict=True) if b == 0]  # the touch measures read the main object
        self.oplanes = np.vstack([p for _v, p, _a, _b in main])
        self.ostarts = np.cumsum([0] + [len(p) for _v, p, _a, _b in main])[:-1]
        self.oends = np.r_[self.ostarts[1:], len(self.oplanes)]
        #: every object piece's world box (n, 3) — _overlap_rows tests a hand piece against all of them at once
        self.olo = np.array([lo for _v, _p, lo, _hi in self.obj]).reshape(-1, 3)
        self.ohi = np.array([hi for _v, _p, _lo, hi in self.obj]).reshape(-1, 3)
        self.touch_idx = sorted({self.names.index(t) for t in touch if t in self.names})
        self.n_eval = 0
        #: _overlap_rows: per (hand piece, object piece), the face axis that gave the pair's depth at its last full
        #: reading — ("o", object plane) or ("h", hand plane)
        self._sep = {}
        #: the palm-relative link placements of the last finger configuration link_T ran FK for: (q bytes, [(R, t)])
        self._fk = (None, None)
        #: per touching link, (its world vertices, a lower bound of each one's signed distance) at the last contact-row
        #: evaluation (_touch_min). The plane values are L-Lipschitz with L = the largest plane normal norm — 1 for a
        #: rotation, other when the frame-0 object quaternion is not unit — so a vertex's bound falls by L × its move
        #: (L rounded up by 1e-12).
        self._lip = float(np.linalg.norm(self.oplanes[:, :3], axis=1).max()) * (1.0 + 1e-12)
        self._lb = {}

    def link_T(self, dq):
        """[(R, t)] world transforms of the strict model's links at dq. The palm-relative placements depend on the
        finger configuration only: the last one's are kept and reused while the configuration's bytes repeat (the
        Jacobian's wrist columns move dq[0:6] alone)."""
        import pinocchio as pin  # noqa: PLC0415 — the f0 interpreter's dependency, loaded on first use

        s = self.s
        q = pin.neutral(s.model)
        for jn, val in zip(s.jnames, s.fingers12(self.indep0 + dq[6:]), strict=True):
            q[s.jidx[jn]] = val
        key = q.tobytes()
        if key != self._fk[0]:
            pin.forwardKinematics(s.model, s.data, q)
            pin.updateFramePlacements(s.model, s.data)
            Tb_inv = s.palm_inv()  # the palm the training robot stands at (HandModel.palm_inv)
            rel = [Tb_inv * s.data.oMf[fid] for fid in self.fids]
            self._fk = (key, [(np.array(r.rotation), np.array(r.translation)) for r in rel])
        wR = _aa_to_R(dq[3:6]) @ self.wR0
        wp = self.wpos0 + dq[:3]
        return [(wR @ R, wR @ t + wp) for R, t in self._fk[1]]

    def sdf(self, x):
        """(signed distance to the cooked hull per point — min over pieces of the max plane value, < 0 inside —, each
        point's nearest piece, the plane values); `normal` reads one point's active plane from the last two."""
        vals = x @ self.oplanes[:, :3].T + self.oplanes[:, 3]
        per = np.maximum.reduceat(vals, self.ostarts, axis=1)
        k = per.argmin(1)
        return per[np.arange(len(x)), k], k, vals

    def _touch_min(self, li, x):
        """(value, index, plane-value row, nearest-piece row) of the vertex of x nearest to the hull — what sdf + argmin
        give — reading only the vertices a carried lower bound cannot rule out. The signed distance is L-Lipschitz
        (self._lip): a vertex's last exact value minus L × how far it moved bounds it from below. The vertex with the
        smallest bound is evaluated first; a vertex whose bound is above that value (+1 nm) cannot be the nearest, and
        one at the same value has a bound at or below it (the first-index tie stays). Evaluated vertices get their
        exact value back."""
        c = self._lb.get(li)
        idx = None
        if c is not None and len(c[0]) == len(x):
            lb = c[1] - self._lip * np.sqrt(((x - c[0]) ** 2).sum(1))
            p0 = int(lb.argmin())
            u = self.sdf(x[p0 : p0 + 1])[0][0]
            idx = np.flatnonzero(lb <= u + 1e-9)
            if len(idx) > SDF_BOUND_FULL * len(x):
                idx = None
        if idx is None:
            s, kp, vals = self.sdf(x)
            idx, nb = np.arange(len(x)), s.copy()
        else:
            s, kp, vals = self.sdf(x[idx])
            nb = lb
            nb[idx] = s
        self._lb[li] = (x.copy(), nb)
        j = int(s.argmin())
        return s[j], int(idx[j]), vals[j : j + 1], kp[j : j + 1]

    def normal(self, vals, k, i: int):
        """The outward normal of point i's active plane: the first maximum of its plane values within its nearest
        piece k[i] (only the point a row is set at needs it)."""
        a, b = self.ostarts[k[i]], self.oends[k[i]]
        return self.oplanes[a + int(vals[i, a:b].argmax()), :3].copy()

    def world(self, T) -> list:
        """Each hand piece's hull vertices in the world at link transforms T (v @ R.T + t, per piece in self.pieces
        order): computed once per `rows` call and read by every row family (the same numbers each would compute)."""
        return [v @ T[li][0].T + T[li][1] for li, v, _e in self.pieces]

    def _link_verts(self, T, li, W=None):
        if W is not None:
            return np.vstack([W[i] for i, (lj, _v, _e) in enumerate(self.pieces) if lj == li])
        R, t = T[li]
        return np.vstack([v @ R.T + t for lj, v, _e in self.pieces if lj == li])

    def _overlap_rows(self, T, r_t, W=None):
        """Rows (residual, link, link-local point, direction that increases the residual) of the overlapping pairs;
        r_t = the overlap target (m) of each object piece; W = `world(T)` (None: computed here). A pair's depth is
        the least over the face axes of both hulls, so one axis bounds it from above: the axis that gave the pair's
        depth at its last full reading (self._sep) is read first, and when it stays SEP_EPS_M below the target the
        pair has no row — the full reading would give none either; any other pair is read in full."""
        if len(r_t) != len(self.obj):
            raise ValueError(f"_overlap_rows: {len(r_t)} targets for {len(self.obj)} object pieces")
        out = []
        for i, (li, v, e) in enumerate(self.pieces):
            R, t = T[li]
            hv = v @ R.T + t if W is None else W[i]
            hlo, hhi = hv.min(0), hv.max(0)
            near = np.flatnonzero(~((hlo > self.ohi).any(1) | (hhi < self.olo).any(1)))  # boxes that meet, in order
            if not len(near):
                continue
            hn = e[:, :3] @ R.T
            hd = e[:, 3] - hn @ t
            for oi in near:
                ov, op, _olo, _ohi = self.obj[oi]
                rt = r_t[oi]
                ax = self._sep.get((i, oi))
                if ax is not None:
                    k, j = ax
                    s = -(hv @ op[j, :3] + op[j, 3]).min() if k == "o" else -(ov @ hn[j] + hd[j]).min()
                    if s <= rt - SEP_EPS_M:
                        continue
                o1 = -(hv @ op[:, :3].T + op[:, 3]).min(0)
                o2 = -(ov @ hn.T + hd).min(0)
                j1, j2 = int(o1.argmin()), int(o2.argmin())
                depth = min(o1[j1], o2[j2])
                self._sep[(i, oi)] = ("o", j1) if o1[j1] <= o2[j2] else ("h", j2)
                if depth <= rt:
                    continue
                if o1[j1] <= o2[j2]:  # object face axis: the deepest hand vertex leaves along the face normal
                    n = op[j1, :3]
                    out.append((depth - rt, li, (hv[int((hv @ n + op[j1, 3]).argmin())] - t) @ R, -n))
                else:  # hand face axis: the hand moves against its face normal, at the deepest object vertex
                    n = hn[j2]
                    out.append((depth - rt, li, (ov[int((ov @ n + hd[j2]).argmin())] - t) @ R, n))
        return out

    def _contact_desk_rows(self, T, W=None):
        """Rows of the touching links that left the hull (beyond OBJ_TOUCH_MM) and of links below the desk floor;
        W = `world(T)` (None: computed here)."""
        out = []
        touch_t = RO.OBJ_TOUCH_MM / 1000.0
        for li in self.touch_idx:
            R, t = T[li]
            hv = self._link_verts(T, li, W)
            sk, k, vrow, krow = self._touch_min(li, hv)
            if sk > touch_t:
                out.append((sk - touch_t, li, (hv[k] - t) @ R, self.normal(vrow, krow, 0)))
        for li in sorted({lj for lj, _v, _e in self.pieces}):
            R, t = T[li]
            hv = self._link_verts(T, li, W)
            k = int(hv[:, 2].argmin())
            if hv[k, 2] < self.floor:
                out.append((self.floor - hv[k, 2], li, (hv[k] - t) @ R, np.array([0.0, 0.0, -1.0])))
        return out

    def rows(self, dq, r_t):
        T = self.link_T(dq)
        W = self.world(T)
        n = len(self.obj)
        out = self._overlap_rows(T, r_t[:n], W)
        for (b, grid), rt in zip(self.stills, r_t[n:], strict=True):
            out += RST.rows(b, grid, self.pieces, self.gates, T, rt, W)
        return out + self._contact_desk_rows(T, W), T

    def jac(self, dq, rows, T):
        Tk = [self.link_T(dq + EPS * np.eye(HC.nd)[k]) for k in range(HC.nd)]
        self.n_eval += HC.nd
        J = np.zeros((len(rows), HC.nd))
        for m, (_r, li, xl, dvec) in enumerate(rows):
            x0 = T[li][0] @ xl + T[li][1]
            for k in range(HC.nd):
                J[m, k] = dvec @ ((Tk[k][li][0] @ xl + Tk[k][li][1]) - x0) / EPS
        return J

    def merit(self, dq, r_t):
        self.n_eval += 1
        rows, T = self.rows(dq, r_t)
        return sum(r * r for r, *_ in rows) + LAMBDA * float(np.sum((HC.w_dq * dq) ** 2)), rows, T

    def clip(self, dq):
        dq = np.asarray(dq, float).copy()
        dq[6:] = np.clip(self.indep0 + dq[6:], self.lo, self.hi) - self.indep0
        return dq

    def solve(self, r_t, dq0=None):
        """Damped Gauss-Newton (Levenberg-Marquardt) from dq0 (None = the data pose) to the overlap targets r_t [m]
        (one per object piece; one number = the same for every piece)."""
        r_t = np.broadcast_to(np.asarray(r_t, float), (len(self.obj) + len(self.stills),))
        dq = self.clip(np.zeros(HC.nd) if dq0 is None else dq0)
        f, rows, T = self.merit(dq, r_t)
        mu, W2 = 1e-3, LAMBDA * HC.w_dq**2
        for _it in range(ITERS):
            if not rows:
                break
            J = self.jac(dq, rows, T)
            r = np.array([x[0] for x in rows])
            A = J.T @ J + np.diag(W2)
            g = J.T @ r + W2 * dq
            ok = False
            for _try in range(10):
                cand = self.clip(dq - np.linalg.solve(A + mu * np.diag(damping_scale(A)), g))
                f2, rows2, T2 = self.merit(cand, r_t)
                if f2 < f:
                    dq, f, rows, T, ok = cand, f2, rows2, T2, True
                    mu = max(mu / 3.0, 1e-9)
                    break
                mu *= 10.0
            if not ok:
                break
        return dq

    def lost_touch(self, dq):
        """Touching links whose nearest vertex is farther than CONTACT_BAND from the cooked hull."""
        T = self.link_T(dq)
        return [self.names[li] for li in self.touch_idx if self.sdf(self._link_verts(T, li))[0].min() > CONTACT_BAND]


def damping_scale(A):
    """The Marquardt scale of each dq: diag(A), except that the wrist translation (dq[0:3]) and rotation (dq[3:6]) take
    the mean of their block (trace / 3). diag(A) itself depends on how the world axes lie: the same hand and object
    turned by the clip's placement then solve to a different answer (measured on oakink2 pour_in_lab 30dbe27 left:
    SAT 9.2 mm on the base tree, 1.8 mm on the placed tree). The trace of a block does not change under a
    rotation of the world, so the answer turns with the problem."""
    d = np.diag(A).astype(float).copy()
    d[0:3] = d[0:3].mean()
    d[3:6] = d[3:6].mean()
    return d + 1e-12


def solve_frame(prob: dict, warm, verify_retries: int):
    """One window frame: retries with a tighter overlap target. prob = {'proj', 'sat_fn', 'desk_fn', 'd0'}; with
    prob['keep_open'] (frame 0), when no attempt passes, the least-SAT attempt that passes the checks at the open
    thresholds is returned and prob['kept'] says so (a later attempt that loses a contact link ends the retries there).
    Returns (dq | None, why, SAT after [mm] of the main object, the result's repair_obj.Sat)."""
    P, sat_fn, desk_fn, d0 = prob["proj"], prob["sat_fn"], prob["desk_fn"], prob["d0"]
    why, s, best = "", None, None
    for attempt in range(max(1, verify_retries + 1)):
        r_t = np.minimum(RO.OBJ_TOUCH_MM, P.res_mm * STRICT_MARGIN_SCALE * 0.5**attempt) / 1000.0
        dq = P.solve(r_t, warm)
        sat = sat_fn(dq)
        s, ln = sat
        lost = P.lost_touch(dq)
        dk = desk_fn(dq)
        if lost:  # a tighter target only pushes further out: no retry
            dropped = ",".join(n.split("hand_")[-1] for n in lost[:4])
            if best is not None:
                break
            return None, f"no result keeps every contact link (dropped: {dropped})", round(s * 1000.0, 2), sat
        failed = RO.failed_checks(s, ln, dk, d0, sat, prob.get("desk_cap"))
        if not failed:
            return dq, "", round(s * 1000.0, 2), sat
        why = f"check failed after {attempt + 1} tries: {failed}"
        if prob.get("keep_open") and not RO.failed_checks(s, ln, dk, d0, sat, prob.get("desk_cap"), k=0):
            if best is None or s < best[2]:
                best = (dq, sat, s, attempt + 1)
    if best is not None:
        dq, sat, s, n = best
        prob["kept"] = (
            f"kept SAT {s * 1000:.2f}mm ≤ {RO.OBJ_REPAIR_MM:g}mm (target {RO.OBJ_RESIDUAL_MM:g}mm not reached, "
            f"attempt {n} of {attempt + 1})"
        )
        return dq, "", round(s * 1000.0, 2), sat
    return None, why, (round(s * 1000.0, 2) if s is not None else None), None


def obj_solve_window(rep: dict, pose0, problems, verify_retries: int, ctr: list, t0: float, desk_start=None) -> dict:
    """The object branch's solve over the window: `problems` = one prob dict per window frame (frame 0 first, built
    lazily by an iterator so a refused frame 0 costs one frame). `desk_start` = (floor, cap) of a desk-deep start
    (repair_obj.desk_start): frame 0 solves with the desk floor there and passes only within the cap. Fills the
    report; returns it."""
    t2 = time.time()
    rows, sats, bodies, carried, prev, kept = [], [], [], [], None, ""
    for f, prob in enumerate(problems):
        if f == 0:
            rep["touch_links_before"] = sorted(n.split("hand_")[-1] for n in prob["touch"])
            if desk_start is not None:
                prob["proj"].floor, prob["desk_cap"] = desk_start
            prob["keep_open"] = True
        dq, why, s, sat = solve_frame(prob, prev, verify_retries)
        ctr[0] += prob["proj"].n_eval
        if dq is None and f == 0:
            _close_report(rep, None, [why], t0, t2)
            return RO.refuse(rep, why)
        if dq is None:
            carried.append(f)
            dq = prev
            sat = prob["sat_fn"](dq)
            s = round(sat[0] * 1000.0, 2)
        if f == 0:
            rep["touch_links_kept"] = len(prob["touch"]) - len(prob["proj"].lost_touch(dq))
            kept = prob.get("kept", "")
        rows.append(dq)
        sats.append(s)
        bodies.append(sat)
        prev = dq
    late = RO.release_failed(sats, bodies)
    if late:
        _close_report(rep, None, [late], t0, t2)
        return RO.refuse(rep, late)
    notes = [f"window {len(rows)} frames · sat ≤ {max(sats):.2f}mm"]
    if kept:  # only a side the window accepts carries the keep (a release refusal above reports its own reason)
        rep["obj_kept"] = kept
        notes.append(kept)
        print(f"{RO.MARK} side={rep.get('side')} frame 0 {kept}", flush=True)
    _close_report(rep, "project", notes, t0, t2)
    _repaired_fields(rep, rows[0])
    rep["_internal"] = (rows[0], *pose0)
    rep["obj_window"] = np.array(rows)
    rep["obj_window_sat_mm"] = sats
    rep["obj_window_carried"] = carried
    rep["obj_sat_after_mm"] = sats[0]
    if getattr(bodies[0], "others", ()):
        rep["obj_window_body_sat_mm"] = {  # each other body's deepest SAT over the window (same bodies every frame)
            n: round(max(b.others[i][2] for b in bodies) * 1000.0, 2) for i, (n, *_r) in enumerate(bodies[0].others)
        }
    W = np.array(rows)
    rep["obj_window_wrist_cm_max"] = round(float(np.linalg.norm(W[:, :3], axis=1).max()) * 100, 2)
    rep["obj_window_step_cm_max"] = (
        round(float(np.linalg.norm(np.diff(W[:, :3], axis=0), axis=1).max()) * 100, 2) if len(W) > 1 else 0.0
    )
    return rep
