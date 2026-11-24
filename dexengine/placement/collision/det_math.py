"""Host-independent float64 arithmetic for the cross-repair decision path — the one owner of its small linear algebra
and trigonometry.

Why: the cross repair measures distances with mj_geomDistance, whose GJK/EPA on penetrating meshes turns a 1-ulp change
of a pose into centimetres (measured: one ulp in the link poses of 4 of 326 frames moved 54 pair distances by up to
19.9 mm), and the greedy turns that into another repair. Three host-dependent sources fed it the ulp:
  - glibc ``sincos`` (pinocchio's FK): 2.35 differs from its own sin/cos in 288 of 400,000 angles, 2.39 does not;
  - OpenBLAS DYNAMIC_ARCH picks a kernel per CPU (Prescott · Haswell · Zen · SkylakeX): ``@`` · ``np.dot`` ·
    ``np.linalg.norm`` (1-D) · ``np.linalg.solve`` change their last bits with it;
  - numpy's AVX512 dispatch (SVML) for ``np.sin`` / ``np.cos`` and friends.
Every function here uses only element-wise float64 ``+ - * /``, ``np.sqrt`` and ``np.rint`` (IEEE correctly rounded,
identical on every x86-64 host and SIMD width — numpy never fuses a multiply and an add across two ufunc calls) in a
fixed order. No BLAS, no libm, no dispatched transcendental. Inputs broadcast over leading axes.

sincos: fdlibm's kernels (__kernel_sin / __kernel_cos, |r| <= pi/4) after a two-term Cody-Waite reduction by pi/2;
error <= 1 ulp for |x| <= 2**20 * pi/2 (larger arguments raise — joint angles never get there).
"""

from __future__ import annotations

import numpy as np

# The sin/cos kernels, the pi/2 reduction and atan below port fdlibm (e_rem_pio2.c · k_sin.c · k_cos.c · s_atan.c):
#   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#   Developed at SunPro, a Sun Microsystems, Inc. business.
#   Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice is
#   preserved.
_INVPIO2 = 6.36619772367581382433e-01
_PIO2_1 = 1.57079632673412561417e00  # first 33 bits of pi/2 — n * _PIO2_1 is exact for |n| < 2**20
_PIO2_1T = 6.07710050650619224932e-11  # pi/2 - _PIO2_1
_S = (
    -1.66666666666666324348e-01,
    8.33333333332248946124e-03,
    -1.98412698298579493134e-04,
    2.75573137070700676789e-06,
    -2.50507602534068634195e-08,
    1.58969099521155010221e-10,
)
_C = (
    4.16666666666666019037e-02,
    -1.38888888887411252052e-03,
    2.48015872894767294178e-05,
    -2.75573143513906633035e-07,
    2.08757232129817482790e-09,
    -1.13596475577881948265e-11,
)
_XMAX = 1.6470993291652855e06  # 2**20 * pi/2


def _ksin(r, z):
    v = z * r
    p = _S[1] + z * (_S[2] + z * (_S[3] + z * (_S[4] + z * _S[5])))
    return r + v * (_S[0] + z * p)


def _kcos(z):
    p = z * (_C[0] + z * (_C[1] + z * (_C[2] + z * (_C[3] + z * (_C[4] + z * _C[5])))))
    hz = 0.5 * z
    w = 1.0 - hz
    return w + (((1.0 - w) - hz) + z * p)


def sincos(x):
    """(sin x, cos x) as float64 arrays of x's shape."""
    x = np.asarray(x, dtype=np.float64)
    if x.size and not np.all(np.abs(x) <= _XMAX):
        raise ValueError(f"[DET_MATH] sincos argument beyond {_XMAX:g} (or not finite): {x[np.abs(x) > _XMAX][:3]}")
    n = np.rint(x * _INVPIO2)
    r = (x - n * _PIO2_1) - n * _PIO2_1T
    z = r * r
    s, c = _ksin(r, z), _kcos(z)
    q = n.astype(np.int64) & 3  # n integral, |n| < 2**20: exact; two's complement gives n mod 4 for n < 0 too
    sn = np.where(q == 0, s, np.where(q == 1, c, np.where(q == 2, -s, -c)))
    cs = np.where(q == 0, c, np.where(q == 1, -s, np.where(q == 2, -c, s)))
    return sn, cs


def dot3(a, b):
    """a · b over the last axis (length 3): (a0 b0 + a1 b1) + a2 b2."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) + a[..., 2] * b[..., 2]


def sumsq(a):
    """Sum of squares over the last axis, left to right."""
    a = np.asarray(a, dtype=np.float64)
    s = a[..., 0] * a[..., 0]
    for i in range(1, a.shape[-1]):
        s = s + a[..., i] * a[..., i]
    return s


def norm(a):
    """Euclidean norm over the last axis (any length): sqrt of sumsq."""
    return np.sqrt(sumsq(a))


def cross3(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        -1,
    )


def matvec3(R, v):
    """R @ v for (..., 3, 3) and (..., 3)."""
    R, v = np.asarray(R, dtype=np.float64), np.asarray(v, dtype=np.float64)
    return (R[..., :, 0] * v[..., None, 0] + R[..., :, 1] * v[..., None, 1]) + R[..., :, 2] * v[..., None, 2]


def matTvec3(R, v):
    """R.T @ v for (..., 3, 3) and (..., 3)."""
    R, v = np.asarray(R, dtype=np.float64), np.asarray(v, dtype=np.float64)
    return (R[..., 0, :] * v[..., None, 0] + R[..., 1, :] * v[..., None, 1]) + R[..., 2, :] * v[..., None, 2]


def matmul3(A, B):
    """A @ B for (..., 3, 3) matrices."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    return (A[..., :, 0, None] * B[..., None, 0, :] + A[..., :, 1, None] * B[..., None, 1, :]) + A[..., :, 2, None] * B[
        ..., None, 2, :
    ]


def trace_AtB(A, B):
    """trace(A.T @ B) = sum_ij A_ij B_ij, rows then columns left to right."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    s = A[..., 0, 0] * B[..., 0, 0]
    for i in range(3):
        for j in range(3):
            if i or j:
                s = s + A[..., i, j] * B[..., i, j]
    return s


def solve3(A, b):
    """x with A x = b for (..., 3, 3) and (..., 3) — Cramer's rule (the cofactor expansion along the first row)."""
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a10, a11, a12 = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    a20, a21, a22 = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    c00 = a11 * a22 - a12 * a21
    c01 = a12 * a20 - a10 * a22
    c02 = a10 * a21 - a11 * a20
    det = (a00 * c00 + a01 * c01) + a02 * c02
    c10 = a02 * a21 - a01 * a22
    c11 = a00 * a22 - a02 * a20
    c12 = a01 * a20 - a00 * a21
    c20 = a01 * a12 - a02 * a11
    c21 = a02 * a10 - a00 * a12
    c22 = a00 * a11 - a01 * a10
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    x0 = (c00 * b0 + c10 * b1) + c20 * b2
    x1 = (c01 * b0 + c11 * b1) + c21 * b2
    x2 = (c02 * b0 + c12 * b1) + c22 * b2
    return np.stack([x0, x1, x2], -1) / det[..., None]


def rot_axis(axis, s, c):
    """Rotation about the unit ``axis`` (3,) by the angle whose sine and cosine are s, c (any shape) → (..., 3, 3):
    c I + s [a]x + (1 - c) a aT."""
    a = np.asarray(axis, dtype=np.float64)
    s, c = np.asarray(s, dtype=np.float64), np.asarray(c, dtype=np.float64)
    t = 1.0 - c
    ax, ay, az = a[0], a[1], a[2]
    R = np.empty(np.shape(c) + (3, 3))
    R[..., 0, 0] = c + t * (ax * ax)
    R[..., 0, 1] = t * (ax * ay) - s * az
    R[..., 0, 2] = t * (ax * az) + s * ay
    R[..., 1, 0] = t * (ax * ay) + s * az
    R[..., 1, 1] = c + t * (ay * ay)
    R[..., 1, 2] = t * (ay * az) - s * ax
    R[..., 2, 0] = t * (ax * az) - s * ay
    R[..., 2, 1] = t * (ay * az) + s * ax
    R[..., 2, 2] = c + t * (az * az)
    return R


def rpy_matrix(rpy) -> np.ndarray:
    """URDF ``rpy`` (roll, pitch, yaw — strings or numbers) → Rz(yaw) · Ry(pitch) · Rx(roll) (3, 3), the product taken
    as (Rz Ry) Rx."""
    r, p, y = (float(v) for v in rpy)
    (sr, sp, sy), (cr, cp, cy) = sincos(np.array([r, p, y]))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return matmul3(matmul3(rz, ry), rx)


def hom(A, B) -> np.ndarray:
    """A · B for 4×4 rigid transforms (last row 0 0 0 1 — not read): rotation A_R B_R, translation A_R b + a."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = matmul3(A[:3, :3], B[:3, :3])
    out[:3, 3] = matvec3(A[:3, :3], B[:3, 3]) + A[:3, 3]
    return out


def mean_rows(a) -> np.ndarray:
    """Mean of the rows of a (n, k) array, summed top to bottom."""
    a = np.asarray(a, dtype=np.float64)
    s = a[0].copy()
    for i in range(1, len(a)):
        s = s + a[i]
    return s / float(len(a))


def acos_deg(c):
    """arccos(c) in degrees for c in [-1, 1] (clipped) — 2 atan2(sqrt(1 - c), sqrt(1 + c)) with fdlibm's atan kernel on
    the ratio, so no libm or dispatched call."""
    c = np.clip(np.asarray(c, dtype=np.float64), -1.0, 1.0)
    y, x = np.sqrt(1.0 - c), np.sqrt(1.0 + c)
    big = y > x  # atan(y/x) = pi/2 - atan(x/y) when y > x keeps the ratio in [0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(big, x / y, y / x)
    t = np.where(np.isfinite(t), t, 0.0)
    a = _atan01(t)
    a = np.where(big, _PIO2_HI - (a - _PIO2_LO), a)
    return (2.0 * a) * _RAD2DEG


# fdlibm s_atan.c kernel on [0, 1]
_PIO2_HI, _PIO2_LO = 1.57079632679489655800e00, 6.12323399573676603587e-17
_RAD2DEG = 57.295779513082320876798
_ATANHI = (4.63647609000806093515e-01, 7.85398163397448278999e-01)
_ATANLO = (2.26987774529616870924e-17, 3.06161699786838301793e-17)
_AT = (
    3.33333333333329318027e-01,
    -1.99999999998764832476e-01,
    1.42857142725034663711e-01,
    -1.11111104054623557880e-01,
    9.09088713343650656196e-02,
    -7.69187620504482999495e-02,
    6.66107313738753120669e-02,
    -5.83357013379057348645e-02,
    4.97687799461593236017e-02,
    -3.65315727442169155270e-02,
    1.62858201153657823623e-02,
)


def _atan_poly(x):
    z = x * x
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    return x, s1, s2


def _atan01(t):
    """atan(t) for t in [0, 1] — fdlibm's three ranges (t < 7/16 · < 11/16 · <= 1)."""
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    r0 = t < 0.4375
    r1 = ~r0 & (t < 0.6875)
    r2 = ~r0 & ~r1
    x, s1, s2 = _atan_poly(t)
    out = np.where(r0, x - x * (s1 + s2), out)
    x1 = (2.0 * t - 1.0) / (2.0 + t)
    x, s1, s2 = _atan_poly(x1)
    out = np.where(r1, _ATANHI[0] - ((x * (s1 + s2) - _ATANLO[0]) - x), out)
    x2 = (t - 1.0) / (t + 1.0)
    x, s1, s2 = _atan_poly(x2)
    out = np.where(r2, _ATANHI[1] - ((x * (s1 + s2) - _ATANLO[1]) - x), out)
    return out


def acos_rad(c):
    """arccos(c) in radians (c clipped to [-1, 1]) — acos_deg's kernel without the degree factor."""
    c = np.clip(np.asarray(c, dtype=np.float64), -1.0, 1.0)
    y, x = np.sqrt(1.0 - c), np.sqrt(1.0 + c)
    big = y > x
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(big, x / y, y / x)
    t = np.where(np.isfinite(t), t, 0.0)
    a = _atan01(t)
    a = np.where(big, _PIO2_HI - (a - _PIO2_LO), a)
    return 2.0 * a


def solve_spd(A, b):
    """x with A x = b for symmetric positive definite (..., n, n) and (..., n) — Cholesky, fixed order."""
    A, b = np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = A.shape[-1]
    L = np.zeros_like(A)
    for j in range(n):
        s = A[..., j, j]
        for k in range(j):
            s = s - L[..., j, k] * L[..., j, k]
        L[..., j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            s = A[..., i, j]
            for k in range(j):
                s = s - L[..., i, k] * L[..., j, k]
            L[..., i, j] = s / L[..., j, j]
    y = np.zeros_like(b)
    for i in range(n):
        s = b[..., i]
        for k in range(i):
            s = s - L[..., i, k] * y[..., k]
        y[..., i] = s / L[..., i, i]
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        s = y[..., i]
        for k in range(i + 1, n):
            s = s - L[..., k, i] * x[..., k]
        x[..., i] = s / L[..., i, i]
    return x
