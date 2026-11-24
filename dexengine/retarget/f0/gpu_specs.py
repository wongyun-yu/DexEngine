"""f0 GPU engine, shared layer: tolerances and engine constants, the Warp import guard, the hand and object specs
uploaded to the device and the wrist rotation rows."""

import numpy as np

# constants (owned here)
HULL_TOL = 1e-12  # half-plane containment tolerance (m)


MAX_BATCH = 8192  # max candidates evaluated in one GPU call (memory, query-point buffers)


# fp32 pre-check guard (m) = FP32_GUARD × (1 + coordinate scale), at least 100× the fp32 conversion error (~1e-7 × |x|).
# Points closer than this to a boundary are re-judged in fp64 (count verdicts equal the fp64 version).
FP32_GUARD = 1e-5


KD_LEAF = 16  # nearest-sample kd-tree leaf size (samples)


KD_BOX_PAD = 1e-6  # fp32 outward padding of kd node boxes (m), so rounding never makes a box miss a point


ENGINE_TAG = "f0_gpu_v4"


# FMA contraction off (Warp module option fuse_fp). When on, NVRTC fuses multiplies and adds at different places per
# GPU generation (sm_89 vs sm_120); fp64 sums then differ by ULPs, which occasionally flips an NM comparison and changes
# an evaluation count on one GPU generation only. Off, the IEEE products and sums follow the written order,
# so the output does not depend on the GPU generation (same purpose as the OPENBLAS_CORETYPE pin).
GPU_FUSE_FP = False


def _require_warp():
    from dexengine.core import runtime_env  # noqa: PLC0415 — warp only when a GPU evaluator is made

    wp = runtime_env.warp()

    wp.config.quiet = True
    wp.init()
    return wp


def _palm_subtree(m, bf):
    """The joints below the joint that parents the palm frame `bf`: (that joint, the subtree joints in model order,
    joint → position in that list)."""
    root = int(bf.parentJoint)
    sub, inside = [], {root}
    for j in range(1, m.njoints):
        if int(m.parents[j]) in inside:
            inside.add(j)
            sub.append(j)
    pos_of = {j: i for i, j in enumerate(sub)}
    return root, sub, pos_of


# hand geometry (FK table + points + hulls)
class GpuHandSpec:
    """One `repair_geom.HandModel` as flat arrays for the GPU. Every value is read from that model (no copies).

    FK origin = the frame of the joint that parents the palm (base_frame); only the joints of the subtree below it are
    followed, which equals the CPU's `Tb_inv * oMf[fid]` (the arm joints cancel). Only 1-DOF revolute joints are
    supported; anything else is fatal."""

    def __init__(self, hand):
        m = hand.model
        bf = m.frames[hand.base_frame]
        root, sub, pos_of = _palm_subtree(m, bf)
        qslot = {int(m.getJointId(jn)): fi for fi, jn in enumerate(hand.jnames)}
        missing = [jn for jn in hand.jnames if int(m.getJointId(jn)) not in pos_of]
        if missing:
            raise SystemExit(f"[f0_gpu] finger joints {missing} are outside the palm subtree")
        self._set_joints(m, sub, pos_of, qslot)
        Pinv = bf.placement.inverse()
        self.palm_inv_R = np.asarray(Pinv.rotation, float)
        self.palm_inv_t = np.asarray(Pinv.translation, float)
        self._set_links(hand, m, root, pos_of)
        self.n_f = len(hand.jnames)

    def _set_joints(self, m, sub, pos_of, qslot):
        """FK joint table of the palm subtree (parent, placement, axis kind, finger slot); 1-DOF revolute only."""
        self.nj = len(sub)
        self.j_parent = np.array([pos_of.get(int(m.parents[j]), -1) for j in sub], np.int32)
        self.j_R = np.zeros((self.nj, 3, 3))
        self.j_t = np.zeros((self.nj, 3))
        self.j_axis = np.zeros((self.nj, 3))
        self.j_kind = np.zeros(self.nj, np.int32)  # 0=RX 1=RY 2=RZ 3=unaligned
        self.j_q = np.full(self.nj, -1, np.int32)
        for i, j in enumerate(sub):
            jm = m.joints[j]
            nm = jm.shortname()
            if nm in ("JointModelRX", "JointModelRY", "JointModelRZ"):
                k = "XYZ".index(nm[-1])
                self.j_kind[i] = k
                self.j_axis[i, k] = 1.0
            elif nm == "JointModelRevoluteUnaligned":
                self.j_kind[i] = 3
                self.j_axis[i] = np.asarray(jm.extract().axis, float)
            else:
                raise SystemExit(
                    f"[f0_gpu] hand subtree joint {m.names[j]} of type {nm} is unsupported (revolute joints only)"
                )
            if jm.nq != 1:
                raise SystemExit(f"[f0_gpu] joint {m.names[j]} nq={jm.nq} is unsupported")
            pl = m.jointPlacements[j]
            self.j_R[i] = pl.rotation
            self.j_t[i] = pl.translation
            self.j_q[i] = qslot.get(j, -1)  # not a finger joint -> neutral 0, same as the CPU's pin.neutral

    def _set_links(self, hand, m, root, pos_of):
        """Per-link frame placement, point ranges and reverse-hull half-planes (with a bounding sphere)."""
        from scipy.spatial import ConvexHull

        L = self.n_links = len(hand.links)
        self.l_joint = np.zeros(L, np.int32)
        self.l_R = np.zeros((L, 3, 3))
        self.l_t = np.zeros((L, 3))
        self.l_hull = np.zeros(L, np.int32)
        self.sph_c = np.zeros((L, 3))
        self.sph_r = np.zeros(L)
        pts, pa, pb, facets, fa, fb = [], [], [], [], [], []
        npt = nf = 0
        for li, (lname, fid, p, dl) in enumerate(hand.links):
            fr = m.frames[fid]
            pj = int(fr.parentJoint)
            if pj == root:
                self.l_joint[li] = -1
            elif pj in pos_of:
                self.l_joint[li] = pos_of[pj]
            else:
                raise SystemExit(f"[f0_gpu] link {lname} is outside the palm subtree")
            self.l_R[li] = fr.placement.rotation
            self.l_t[li] = fr.placement.translation
            p = np.asarray(p, float)
            pts.append(p)
            pa.append(npt)
            npt += len(p)
            pb.append(npt)
            fa.append(nf)
            if dl is not None:
                P = np.asarray(dl.points, float)  # = shrunk (the Delaunay input as is)
                eq = ConvexHull(P).equations
                facets.append(eq)
                nf += len(eq)
                c = P.mean(0)
                self.sph_c[li] = c
                self.sph_r[li] = np.sqrt(((P - c) ** 2).sum(1)).max()
                self.l_hull[li] = 1
            fb.append(nf)
        self.pts = np.vstack(pts)
        self.pt_a = np.asarray(pa, np.int32)
        self.pt_b = np.asarray(pb, np.int32)
        self.facets = np.vstack(facets) if facets else np.zeros((0, 4))
        self.f_a = np.asarray(fa, np.int32)
        self.f_b = np.asarray(fb, np.int32)
        self.n_pts = len(self.pts)


class GpuObjectSpec:
    """One `repair_geom.ObjectGeom` as flat arrays (samples, normals, hull half-planes, AABB, DOP, reverse surface
    points)."""

    def __init__(self, obj):
        from scipy.spatial import ConvexHull

        self.s_pts = np.ascontiguousarray(obj._s_pts, float)
        self.s_nrm = np.ascontiguousarray(obj._s_nrm, float)
        self.hull_eq = ConvexHull(np.asarray(obj.delaunay_local.points, float)).equations
        self.lo = np.asarray(obj._lo, float)
        self.hi = np.asarray(obj._hi, float)
        self.dop_D = np.asarray(obj._dop_D, float)
        self.dop_hi = np.asarray(obj._dop_hi, float)
        self.local_surface = np.asarray(obj.local_surface, float)
        self._build_kd()

    def _build_kd(self):
        """Sample kd-tree (pre-order, escape pointers). Node boxes are padded outward in fp32 so pruning is
        conservative."""
        P = self.s_pts
        lo_l, hi_l, left, right, a_l, b_l, perm = [], [], [], [], [], [], []

        def build(idx):
            i = len(lo_l)
            lo_l.append(None)
            hi_l.append(None)
            left.append(-1)
            right.append(-1)
            a_l.append(0)
            b_l.append(0)
            p = P[idx]
            lo, hi = p.min(0), p.max(0)
            lo_l[i] = np.nextafter((lo - KD_BOX_PAD).astype(np.float32), np.float32(-np.inf))
            hi_l[i] = np.nextafter((hi + KD_BOX_PAD).astype(np.float32), np.float32(np.inf))
            if len(idx) <= KD_LEAF:
                a_l[i] = len(perm)
                perm.extend(int(x) for x in idx)
                b_l[i] = len(perm)
                return i
            ax = int(np.argmax(hi - lo))
            m = len(idx) // 2
            order = np.argpartition(p[:, ax], m, kind="introselect")
            left[i] = build(idx[order[:m]])
            right[i] = build(idx[order[m:]])
            return i

        build(np.arange(len(P)))
        n = len(lo_l)
        size = np.ones(n, np.int64)
        for i in range(n - 1, -1, -1):  # pre-order: child index > parent
            if left[i] >= 0:
                size[i] = 1 + size[left[i]] + size[right[i]]
        miss = np.array([i + size[i] if i + size[i] < n else -1 for i in range(n)], np.int32)
        self.kd_lo, self.kd_hi = np.stack(lo_l).astype(np.float32), np.stack(hi_l).astype(np.float32)
        self.kd_left, self.kd_miss = np.asarray(left, np.int32), miss
        self.kd_right = np.asarray(right, np.int32)
        self.kd_a, self.kd_b = np.asarray(a_l, np.int32), np.asarray(b_l, np.int32)
        self.kd_perm = np.asarray(perm, np.int32)
        assert sorted(self.kd_perm.tolist()) == list(range(len(P))), "kd permutation does not cover all samples"


# evaluator
def wrist_R_rows(V, W0):
    """Rows bit-identical to `repair._aa_to_R(v) @ W0` (measured on 20,000 rows: 100 %).

    Uses the same primitive operations in the same path: θ per row via np.linalg.norm (BLAS ddot; an einsum sum differed
    by 1 ULP on 12 % of rows), K@K and R@W0 as per-row 2D matmuls, sin, cos and element-wise ops vectorised (identical
    to per row). Rows with θ < 1e-12 use I on the CPU, and I@W0 = W0 exactly, so W0 is used as is (all finger classes
    take this path)."""
    V = np.asarray(V, float)
    W0 = np.asarray(W0, float)
    out = W0.copy()
    cand = np.flatnonzero(np.any(V != 0.0, axis=1))  # zero vector -> θ = 0 -> I (no norm needed)
    if cand.size == 0:
        return out
    thc = np.array([np.linalg.norm(v) for v in V[cand]])
    keep = thc >= 1e-12
    idx, th = cand[keep], thc[keep]
    if idx.size == 0:
        return out
    Vm = V[idx]
    k = Vm / th[:, None]
    n = len(idx)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    KK = np.array([Ki @ Ki for Ki in K])
    R = np.eye(3) + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * KK
    out[idx] = np.array([Ri @ Wi for Ri, Wi in zip(R, W0[idx], strict=True)])
    return out
