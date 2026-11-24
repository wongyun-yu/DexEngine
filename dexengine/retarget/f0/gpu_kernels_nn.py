"""f0 GPU engine, second Warp module: nearest object samples and hull containment (nn), the reverse check
(rev: object points in link hulls), the per-candidate reduction (red). Imported by gpu_kernels; fuse_fp is set
here as well, before the first definition."""
# ruff: noqa: UP018  (int(0) inside the Warp kernels makes a mutable typed local; a bare 0 would be a constant)

from dexengine.core import runtime_env
from dexengine.retarget.f0.gpu_specs import GPU_FUSE_FP

wp = runtime_env.warp()
wp.set_module_options({"fuse_fp": GPU_FUSE_FP})
f64 = wp.float64
v3 = wp.vec3d
m33 = wp.mat33d


@wp.func
def boxd2(q: wp.vec3f, lo: wp.vec3f, hi: wp.vec3f) -> wp.float32:
    dx = wp.max(wp.max(lo[0] - q[0], q[0] - hi[0]), wp.float32(0.0))
    dy = wp.max(wp.max(lo[1] - q[1], q[1] - hi[1]), wp.float32(0.0))
    dz = wp.max(wp.max(lo[2] - q[2], q[2] - hi[2]), wp.float32(0.0))
    return dx * dx + dy * dy + dz * dz


@wp.func
def hull_in(
    q: v3,
    qf: wp.vec3f,
    ho: int,
    hn: int,
    h_n: wp.array(dtype=v3),
    h_d: wp.array(dtype=f64),
    h_n32: wp.array(dtype=wp.vec3f),
    h_d32: wp.array(dtype=wp.float32),
    htol: f64,
    g: wp.float32,
) -> int:
    for f in range(ho, ho + hn):
        v32 = wp.dot(h_n32[f], qf) + h_d32[f]
        if v32 > g:
            return 0  # certainly outside
        if v32 > -g:
            if wp.dot(h_n[f], q) + h_d[f] > htol:  # ambiguous -> fp64
                return 0
    return 1


@wp.func
def leaf_scan(
    a: int,
    b: int,
    po: int,
    so: int,
    kd_perm: wp.array(dtype=wp.int32),
    s32: wp.array(dtype=wp.vec3f),
    s64: wp.array(dtype=v3),
    q: v3,
    qf: wp.vec3f,
    tol: wp.float32,
    b64_in: f64,
    bi_in: int,
    best32_in: wp.float32,
    bnd_in: wp.float32,
):
    # One kd leaf (perm entries a..b): samples within the fp32 bound are re-measured in fp64 (argmin, ties -> lowest
    # index); the bound shrinks to (√best32 + tol)². Returns the updated (b64, bi, best32, bnd).
    b64 = f64(b64_in)
    bi = int(bi_in)
    best32 = wp.float32(best32_in)
    bnd = wp.float32(bnd_in)
    for kk in range(a, b):
        j = kd_perm[po + kk]
        d = s32[so + j] - qf
        d2 = wp.dot(d, d)
        if d2 <= bnd:
            dd = q - s64[so + j]
            e2 = dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]
            if e2 < b64 or (e2 == b64 and j < bi):
                b64 = e2
                bi = j
            if d2 < best32:
                best32 = d2
                rb = wp.sqrt(best32) + tol
                bnd = rb * rb
    return b64, bi, best32, bnd


@wp.func
def nearest_sample(
    q: v3,
    qf: wp.vec3f,
    tol: wp.float32,
    so: int,
    ko: int,
    po: int,
    kd_lo: wp.array(dtype=wp.vec3f),
    kd_hi: wp.array(dtype=wp.vec3f),
    kd_left: wp.array(dtype=wp.int32),
    kd_right: wp.array(dtype=wp.int32),
    kd_miss: wp.array(dtype=wp.int32),
    kd_a: wp.array(dtype=wp.int32),
    kd_b: wp.array(dtype=wp.int32),
    kd_perm: wp.array(dtype=wp.int32),
    s32: wp.array(dtype=wp.vec3f),
    s64: wp.array(dtype=v3),
):
    # Nearest = fp64 argmin (ties -> lowest index). kd-tree pruning (fp32 boxes): drop a node whose box distance
    # exceeds √best32 + tol. tol = the per-problem guard g (at least 100× the fp32 error), so the node holding the
    # true nearest sample is never dropped; inside a leaf, samples are filtered in fp32 and re-measured in fp64. The
    # result equals an exhaustive fp64 search (not an approximation). Returns (squared distance, sample index).
    b64 = f64(1.0e300)
    bi = int(-1)
    best32 = wp.float32(3.0e38)
    # 1) descend to the home leaf first to get an upper bound
    nd = int(0)
    while kd_left[ko + nd] >= 0:
        lidx = kd_left[ko + nd]
        r = kd_right[ko + nd]
        if boxd2(qf, kd_lo[ko + lidx], kd_hi[ko + lidx]) <= boxd2(qf, kd_lo[ko + r], kd_hi[ko + r]):
            nd = lidx
        else:
            nd = r
    bnd = wp.float32(3.0e38)
    b64, bi, best32, bnd = leaf_scan(
        kd_a[ko + nd], kd_b[ko + nd], po, so, kd_perm, s32, s64, q, qf, tol, b64, bi, best32, bnd
    )
    # 2) pre-order traversal (escape pointers): skip a whole branch when its box is beyond the bound
    i = int(0)
    while i >= 0:
        if boxd2(qf, kd_lo[ko + i], kd_hi[ko + i]) > bnd:
            i = kd_miss[ko + i]
            continue
        if kd_left[ko + i] >= 0:
            i = i + 1
            continue
        b64, bi, best32, bnd = leaf_scan(
            kd_a[ko + i], kd_b[ko + i], po, so, kd_perm, s32, s64, q, qf, tol, b64, bi, best32, bnd
        )
        i = kd_miss[ko + i]
    return b64, bi


@wp.kernel
def nn_kernel(
    q_k: wp.array(dtype=wp.int32),
    q_loc: wp.array(dtype=v3),
    c_prob: wp.array(dtype=wp.int32),
    p_obj: wp.array(dtype=wp.int32),
    p_clr: wp.array(dtype=f64),
    p_pm: wp.array(dtype=f64),
    p_g: wp.array(dtype=wp.float32),
    o_s_off: wp.array(dtype=wp.int32),
    o_s_n: wp.array(dtype=wp.int32),
    o_kd_off: wp.array(dtype=wp.int32),
    o_kp_off: wp.array(dtype=wp.int32),
    kd_lo: wp.array(dtype=wp.vec3f),
    kd_hi: wp.array(dtype=wp.vec3f),
    kd_left: wp.array(dtype=wp.int32),
    kd_right: wp.array(dtype=wp.int32),
    kd_miss: wp.array(dtype=wp.int32),
    kd_a: wp.array(dtype=wp.int32),
    kd_b: wp.array(dtype=wp.int32),
    kd_perm: wp.array(dtype=wp.int32),
    s32: wp.array(dtype=wp.vec3f),
    s64: wp.array(dtype=v3),
    n64: wp.array(dtype=v3),
    o_h_off: wp.array(dtype=wp.int32),
    o_h_n: wp.array(dtype=wp.int32),
    h_n: wp.array(dtype=v3),
    h_d: wp.array(dtype=f64),
    h_n32: wp.array(dtype=wp.vec3f),
    h_d32: wp.array(dtype=wp.float32),
    htol: f64,
    r_cnt: wp.array(dtype=wp.int32),
    r_deep: wp.array(dtype=f64),
    r_close: wp.array(dtype=f64),
):
    e = wp.tid()
    k = q_k[e]
    pr = c_prob[k]
    o = p_obj[pr]
    q = q_loc[e]
    qf = wp.vec3f(wp.float32(q[0]), wp.float32(q[1]), wp.float32(q[2]))
    so = o_s_off[o]
    g = p_g[pr]
    b64, bi = nearest_sample(
        q, qf, g, so, o_kd_off[o], o_kp_off[o], kd_lo, kd_hi, kd_left, kd_right, kd_miss, kd_a, kd_b, kd_perm, s32, s64
    )
    dist = wp.sqrt(b64)
    inside = int(0)
    if hull_in(q, qf, o_h_off[o], o_h_n[o], h_n, h_d, h_n32, h_d32, htol, g) == 1:
        sg = q - s64[so + bi]
        nn = n64[so + bi]
        if sg[0] * nn[0] + sg[1] * nn[1] + sg[2] * nn[2] < f64(0.0):
            inside = 1
    clr = p_clr[pr]
    c = int(0)
    vd = f64(0.0)
    vc = f64(0.0)
    if inside == 1 and dist > p_pm[pr]:
        c = 1
        vd = dist
    if inside == 0 and dist < clr:
        c = 1
        vc = clr - dist
    r_cnt[e] = c
    r_deep[e] = vd
    r_close[e] = vc


@wp.kernel
def rev_kernel(
    c_prob: wp.array(dtype=wp.int32),
    c_hand: wp.array(dtype=wp.int32),
    L_R: wp.array2d(dtype=m33),
    L_t: wp.array2d(dtype=v3),
    L_R32: wp.array2d(dtype=wp.mat33f),
    L_t32: wp.array2d(dtype=wp.vec3f),
    h_l_off: wp.array(dtype=wp.int32),
    h_nl: wp.array(dtype=wp.int32),
    l_hull: wp.array(dtype=wp.int32),
    f_a: wp.array(dtype=wp.int32),
    f_b: wp.array(dtype=wp.int32),
    f_n: wp.array(dtype=v3),
    f_d: wp.array(dtype=f64),
    f_n32: wp.array(dtype=wp.vec3f),
    f_d32: wp.array(dtype=wp.float32),
    sph_c32: wp.array(dtype=wp.vec3f),
    sph_r: wp.array(dtype=f64),
    p_rev_off: wp.array(dtype=wp.int32),
    p_rev_n: wp.array(dtype=wp.int32),
    rev_pts: wp.array(dtype=v3),
    rev32: wp.array(dtype=wp.vec3f),
    p_g: wp.array(dtype=wp.float32),
    htol: f64,
    rev_cnt: wp.array(dtype=wp.int32),
):
    k, lidx = wp.tid()
    h = c_hand[k]
    if lidx >= h_nl[h]:
        return
    li = h_l_off[h] + lidx
    if l_hull[li] == 0:
        return
    pr = c_prob[k]
    RT = wp.transpose(L_R[k, lidx])
    t = L_t[k, lidx]
    RT32 = wp.transpose(L_R32[k, lidx])
    t32 = L_t32[k, lidx]
    g = p_g[pr]
    c32 = sph_c32[li]
    rr = wp.float32(sph_r[li]) + g  # bounding sphere + guard: outside = surely outside
    rr2 = rr * rr
    a = f_a[li]
    b = f_b[li]
    ro = p_rev_off[pr]
    n = int(0)
    for s in range(p_rev_n[pr]):
        loc32 = RT32 * (rev32[ro + s] - t32)
        dv = loc32 - c32
        if wp.dot(dv, dv) > rr2:
            continue
        loc = RT * (rev_pts[ro + s] - t)
        ins = int(1)
        for f in range(a, b):
            v32 = wp.dot(f_n32[f], loc32) + f_d32[f]
            if v32 > g:
                ins = 0
                break
            if v32 > -g:
                if wp.dot(f_n[f], loc) + f_d[f] > htol:
                    ins = 0
                    break
        n += ins
    if n > 0:
        wp.atomic_add(rev_cnt, k, n)


@wp.kernel
def reduce_kernel(
    Lmax: int,
    n_cand: int,
    n_q: int,
    desk_sum: wp.array(dtype=f64),
    desk_cnt: wp.array(dtype=wp.int32),
    q_off: wp.array(dtype=wp.int32),
    r_cnt: wp.array(dtype=wp.int32),
    r_deep: wp.array(dtype=f64),
    r_close: wp.array(dtype=f64),
    rev_cnt: wp.array(dtype=wp.int32),
    out_cnt: wp.array(dtype=wp.int32),
    out_depth: wp.array(dtype=f64),
):
    k = wp.tid()
    ds = f64(0.0)
    dc = int(0)
    for lidx in range(Lmax):
        ds += desk_sum[k * Lmax + lidx]
        dc += desk_cnt[k * Lmax + lidx]
    a = q_off[k * Lmax]
    b = n_q
    if k + 1 < n_cand:
        b = q_off[(k + 1) * Lmax]
    sd = f64(0.0)
    sc = f64(0.0)
    oc = int(0)
    for e in range(a, b):
        oc += r_cnt[e]
        sd += r_deep[e]
        sc += r_close[e]
    out_cnt[k] = dc + oc + rev_cnt[k]
    out_depth[k] = ds + (sd + sc)
