"""f0 GPU engine: the Warp kernels — forward kinematics (fk) and penetration points (pts) here, nearest samples,
reverse checks and the reduction in gpu_kernels_nn. Importing this module imports warp and defines the kernels once
per process (fuse_fp is set in each Warp module before its first definition). The one importer,
gpu_eval.GpuEvaluator, imports it lazily, so the f0 modules load without warp until a GPU evaluator is made."""
# ruff: noqa: UP018  (int(0) inside the Warp kernels makes a mutable typed local; a bare 0 would be a constant)

from dexengine.core import runtime_env
from dexengine.retarget.f0.gpu_kernels_nn import nn_kernel, reduce_kernel, rev_kernel
from dexengine.retarget.f0.gpu_specs import GPU_FUSE_FP

wp = runtime_env.warp()
wp.set_module_options({"fuse_fp": GPU_FUSE_FP})
f64 = wp.float64
v3 = wp.vec3d
m33 = wp.mat33d


@wp.func
def rot_of(kind: int, ax: v3, q: f64) -> m33:
    c = wp.cos(q)
    s = wp.sin(q)
    one = f64(1.0)
    zero = f64(0.0)
    if kind == 0:
        return m33(one, zero, zero, zero, c, -s, zero, s, c)
    if kind == 1:
        return m33(c, zero, s, zero, one, zero, -s, zero, c)
    if kind == 2:
        return m33(c, -s, zero, s, c, zero, zero, zero, one)
    t = one - c  # pinocchio toRotationMatrix(axis, cos, sin)
    x = ax[0]
    y = ax[1]
    z = ax[2]
    return m33(
        c + t * x * x,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        c + t * y * y,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        c + t * z * z,
    )


@wp.kernel
def fk_kernel(
    c_hand: wp.array(dtype=wp.int32),
    c_wp: wp.array(dtype=v3),
    c_wR: wp.array(dtype=m33),
    c_f: wp.array2d(dtype=f64),
    h_j_off: wp.array(dtype=wp.int32),
    h_nj: wp.array(dtype=wp.int32),
    j_parent: wp.array(dtype=wp.int32),
    j_R: wp.array(dtype=m33),
    j_t: wp.array(dtype=v3),
    j_axis: wp.array(dtype=v3),
    j_kind: wp.array(dtype=wp.int32),
    j_q: wp.array(dtype=wp.int32),
    h_l_off: wp.array(dtype=wp.int32),
    h_nl: wp.array(dtype=wp.int32),
    l_joint: wp.array(dtype=wp.int32),
    l_R: wp.array(dtype=m33),
    l_t: wp.array(dtype=v3),
    h_pinvR: wp.array(dtype=m33),
    h_pinvt: wp.array(dtype=v3),
    s_R: wp.array2d(dtype=m33),
    s_t: wp.array2d(dtype=v3),
    out_R: wp.array2d(dtype=m33),
    out_t: wp.array2d(dtype=v3),
    out_R32: wp.array2d(dtype=wp.mat33f),
    out_t32: wp.array2d(dtype=wp.vec3f),
):
    k = wp.tid()
    h = c_hand[k]
    jo = h_j_off[h]
    for i in range(h_nj[h]):
        p = j_parent[jo + i]
        qi = j_q[jo + i]
        q = f64(0.0)
        if qi >= 0:
            q = c_f[k, qi]
        Rj = j_R[jo + i] * rot_of(j_kind[jo + i], j_axis[jo + i], q)
        tj = j_t[jo + i]
        if p >= 0:
            s_R[k, i] = s_R[k, p] * Rj
            s_t[k, i] = s_R[k, p] * tj + s_t[k, p]
        else:
            s_R[k, i] = Rj
            s_t[k, i] = tj
    lo = h_l_off[h]
    PR = h_pinvR[h]
    Pt = h_pinvt[h]
    WR = c_wR[k]
    Wt = c_wp[k]
    for lidx in range(h_nl[h]):
        jl = l_joint[lo + lidx]
        R = l_R[lo + lidx]
        t = l_t[lo + lidx]
        if jl >= 0:
            t = s_R[k, jl] * t + s_t[k, jl]
            R = s_R[k, jl] * R
        oR_ = WR * (PR * R)
        ot_ = WR * (PR * t + Pt) + Wt
        out_R[k, lidx] = oR_
        out_t[k, lidx] = ot_
        out_R32[k, lidx] = wp.mat33f(
            wp.float32(oR_[0, 0]),
            wp.float32(oR_[0, 1]),
            wp.float32(oR_[0, 2]),
            wp.float32(oR_[1, 0]),
            wp.float32(oR_[1, 1]),
            wp.float32(oR_[1, 2]),
            wp.float32(oR_[2, 0]),
            wp.float32(oR_[2, 1]),
            wp.float32(oR_[2, 2]),
        )
        out_t32[k, lidx] = wp.vec3f(wp.float32(ot_[0]), wp.float32(ot_[1]), wp.float32(ot_[2]))


# fp32 pre-check rule
# fp32 decides only the certain side: it rejects or accepts only when a value is more than g (per-problem guard,
# FP32_GUARD × (1 + coordinate scale)) from the boundary; otherwise fp64 redoes the CPU comparison. Count verdicts
# therefore equal the fp64 version.
@wp.func
def aabb32_out(loc: wp.vec3f, lo: wp.vec3f, hi: wp.vec3f, lim: wp.float32) -> int:
    if loc[0] < lo[0] - lim or loc[0] > hi[0] + lim:
        return 1
    if loc[1] < lo[1] - lim or loc[1] > hi[1] + lim:
        return 1
    if loc[2] < lo[2] - lim or loc[2] > hi[2] + lim:
        return 1
    return 0


@wp.func
def aabb64_in(loc: v3, lo: v3, hi: v3, clr: f64) -> int:
    # same comparison as the CPU (sample AABB ± clr, x -> y·z)
    if loc[0] < lo[0] - clr or loc[0] > hi[0] + clr:
        return 0
    if loc[1] < lo[1] - clr or loc[1] > hi[1] + clr:
        return 0
    if loc[2] < lo[2] - clr or loc[2] > hi[2] + clr:
        return 0
    return 1


@wp.func
def dop32_out(
    loc: wp.vec3f, dop_D: wp.array(dtype=wp.vec3f), dop_hi: wp.array(dtype=wp.float32), do: int, lim: wp.float32
) -> int:
    # The 26-DOP is a rejection filter that never changes the result (outside the DOP by > clr => outside the hull,
    # and > clr from every sample => neither deep nor close), so fp32 only rejects, generously (lim = clr + g); the
    # points kept are judged exactly below.
    for d in range(26):
        if wp.dot(dop_D[do + d], loc) - dop_hi[do + d] > lim:
            return 1
    return 0


@wp.struct
class HandPoints:
    """The link point clouds of every registered hand: per hand the first link and the link count, per link its point
    range (relative to the hand's first point), and the points in fp64 and fp32."""

    h_l_off: wp.array(dtype=wp.int32)
    h_nl: wp.array(dtype=wp.int32)
    h_p_off: wp.array(dtype=wp.int32)
    pt_a: wp.array(dtype=wp.int32)
    pt_b: wp.array(dtype=wp.int32)
    pts: wp.array(dtype=v3)
    pts32: wp.array(dtype=wp.vec3f)


@wp.struct
class ProblemFrames:
    """Per problem slot: its object, the object pose (fp64 and fp32), the desk height, the clearance and the guard."""

    p_obj: wp.array(dtype=wp.int32)
    p_opos: wp.array(dtype=v3)
    p_oR: wp.array(dtype=m33)
    p_opos32: wp.array(dtype=wp.vec3f)
    p_oR32: wp.array(dtype=wp.mat33f)
    p_floor: wp.array(dtype=f64)
    p_clr: wp.array(dtype=f64)
    p_g: wp.array(dtype=wp.float32)


@wp.struct
class ObjectBounds:
    """Per object: the sample AABB (fp64 and fp32) and the 26-DOP (26 directions and their maxima per object)."""

    o_lo: wp.array(dtype=v3)
    o_hi: wp.array(dtype=v3)
    o_lo32: wp.array(dtype=wp.vec3f)
    o_hi32: wp.array(dtype=wp.vec3f)
    dop_D32: wp.array(dtype=wp.vec3f)
    dop_hi32: wp.array(dtype=wp.float32)


def point_pass_structs(g: dict) -> tuple:
    """pts_kernel's struct arguments (HandPoints, ProblemFrames, ObjectBounds) over the evaluator's device arrays."""
    out = []
    for cls in (HandPoints, ProblemFrames, ObjectBounds):
        st = cls()
        for name in cls.vars:
            setattr(st, name, g[name])
        out.append(st)
    return tuple(out)


@wp.kernel
def pts_kernel(
    write: int,
    Lmax: int,
    c_prob: wp.array(dtype=wp.int32),
    c_hand: wp.array(dtype=wp.int32),
    L_R: wp.array2d(dtype=m33),
    L_t: wp.array2d(dtype=v3),
    L_R32: wp.array2d(dtype=wp.mat33f),
    L_t32: wp.array2d(dtype=wp.vec3f),
    hand: HandPoints,
    prob: ProblemFrames,
    obj: ObjectBounds,
    desk_sum: wp.array(dtype=f64),
    desk_cnt: wp.array(dtype=wp.int32),
    q_cnt: wp.array(dtype=wp.int32),
    q_off: wp.array(dtype=wp.int32),
    q_k: wp.array(dtype=wp.int32),
    q_loc: wp.array(dtype=v3),
):
    k, lidx = wp.tid()
    h = c_hand[k]
    if lidx >= hand.h_nl[h]:
        if write == 0:
            q_cnt[k * Lmax + lidx] = 0
            desk_sum[k * Lmax + lidx] = f64(0.0)
            desk_cnt[k * Lmax + lidx] = 0
        return
    pr = c_prob[k]
    R = L_R[k, lidx]
    t = L_t[k, lidx]
    R32 = L_R32[k, lidx]
    t32 = L_t32[k, lidx]
    o = prob.p_obj[pr]
    opos = prob.p_opos[pr]
    oRT = wp.transpose(prob.p_oR[pr])
    opos32 = prob.p_opos32[pr]
    oRT32 = wp.transpose(prob.p_oR32[pr])
    fl = prob.p_floor[pr]
    clr = prob.p_clr[pr]
    lim = wp.float32(clr) + prob.p_g[pr]
    lo = obj.o_lo[o]
    hi = obj.o_hi[o]
    lo32 = obj.o_lo32[o]
    hi32 = obj.o_hi32[o]
    li = hand.h_l_off[h] + lidx
    base = hand.h_p_off[h]
    ds = f64(0.0)
    dc = int(0)
    nq = int(0)
    w0 = int(0)
    if write == 1:
        w0 = q_off[k * Lmax + lidx]
    r20 = R[2, 0]
    r21 = R[2, 1]
    r22 = R[2, 2]
    t2 = t[2]
    for i in range(hand.pt_a[li], hand.pt_b[li]):
        p = hand.pts[base + i]
        if write == 0:
            z = r20 * p[0] + r21 * p[1] + r22 * p[2] + t2  # desk: fp64 (same comparison as the CPU)
            if z < fl:
                dc += 1
            ds += wp.max(fl - z, f64(0.0))
        loc32 = oRT32 * (R32 * hand.pts32[base + i] + t32 - opos32)
        if aabb32_out(loc32, lo32, hi32, lim) == 1:
            continue
        if dop32_out(loc32, obj.dop_D32, obj.dop_hi32, o * 26, lim + wp.float32(1.0e-9)) == 1:
            continue
        loc = oRT * (R * p + t - opos)  # fp64 for candidates only
        if aabb64_in(loc, lo, hi, clr) == 0:
            continue
        if write == 1:
            q_k[w0 + nq] = k
            q_loc[w0 + nq] = loc
        nq += 1
    if write == 0:
        q_cnt[k * Lmax + lidx] = nq
        desk_sum[k * Lmax + lidx] = ds
        desk_cnt[k * Lmax + lidx] = dc


#: the launch table GpuEvaluator reads (gpu_eval: self.K)
KERNELS = {"fk": fk_kernel, "pts": pts_kernel, "nn": nn_kernel, "rev": rev_kernel, "red": reduce_kernel}
#: Warp kernels (code — deepcopy cannot copy them): a resident worker keeps the table as it is
#: (placement/bake/server.py rule 2)
_SERVER_KEEP = ("KERNELS",)
