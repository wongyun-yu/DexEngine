"""Stateless helpers of the contact derivation (dexengine/data/contact/derive.py): the rigid
placement transform, the stored time map, residual gates and the ledger merge.

Name-preserving split: moved verbatim from dexengine/data/contact/derive.py, which re-exports every name.
The state that is rebound at run time (the configured roots, the lazy module caches) stays in
dexengine/data/contact/derive.py and nothing here reads it. Imported through dexengine/data/contact/derive.py, which
puts scripts/placement on sys.path first (for `paths`). REPO is this file's own repository root (same directory as
dexengine/data/contact/derive.py, so the same path).
"""

from __future__ import annotations

import fcntl
import os
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core.paths import data_rel, mode_record

REPO = Path(__file__).resolve().parents[3]

# The data-root rebasing that lived here (`_DATA_ROOT = real_repo_root()` + `_rebase`) is gone:
# the layer owner (paths.layer_path) now roots **every** layer — data/ and precompute/ — at the data root
# main_root(), so a second re-rooting step had nothing left to do but undo an isolated output root
# (the sandbox of a regress run).
# Paths come from the owner as is.

DT = 0.02

# Residual threshold: an exact synthesis is two composed float64 linear interpolations, so the residual is at
# **machine precision** (measured max 3e-10 mm). 0.05 mm is a safety line 8 orders looser — one frame of
# misalignment already shows up as the object's per-frame motion (mm scale), so even a generous line catches it.
RESID_MAX_MM = 0.05
# Discriminative power: if shifting the map by 2 frames leaves a residual below this value, the object is
# effectively at rest and G2 would pass any map → WEAK_GATE.
DISCRIM_MIN_MM = 0.5

BOOL_KEYS = ("contact", "contact_bin", "penetration", "palm_penetration")
NON_TIME_KEYS = ("fingers", "regions", "thresholds_m")

LEDGER_COLS = [
    "clip",
    "side",
    "strategy",
    "order",
    "T_base",
    "T_pre",
    "P",
    "tail",
    "T_fin",
    "resid_p50_mm",
    "resid_max_mm",
    "discrim_mm",
    "srcidx_dev_f",
    "map_first",
    "map_last",
    "mesh_canon",
    "mesh_tree",
    "slot_tree",
    "contact_frames_base",
    "contact_frames_out",
    "contact_frac_out",
    "pen_frames_out",
    "and_dropped",
    "status",
]


# ────────────────────────────── placement (rigid) transform ──────────────────────────────


def _placement(ti: dict):
    dx, dy = ti.get("xy_shift_traj", [0.0, 0.0])
    yaw = float(ti.get("yaw_deg", 0.0))
    piv = ti.get("pivot_traj", [0.0, 0.0, 0.0])
    return float(dx), float(dy), yaw, (float(piv[0]), float(piv[1]))


def _fwd(p: np.ndarray, dx, dy, yaw, piv) -> np.ndarray:
    """Same formula as apply_xy_shift.rt_pts: (p − pivot) @ Rz.T + pivot + d."""
    if yaw == 0.0 and dx == 0.0 and dy == 0.0:
        return p
    Rz = Rot.from_euler("z", np.deg2rad(yaw)).as_matrix()
    piv3 = np.array([piv[0], piv[1], 0.0])
    d3 = np.array([dx, dy, 0.0])
    shp = p.shape
    return (((p.reshape(-1, 3) - piv3) @ Rz.T) + piv3 + d3).reshape(shp)


def _inv(p: np.ndarray, dx, dy, yaw, piv) -> np.ndarray:
    if yaw == 0.0 and dx == 0.0 and dy == 0.0:
        return p
    Rz = Rot.from_euler("z", np.deg2rad(yaw)).as_matrix()
    piv3 = np.array([piv[0], piv[1], 0.0])
    d3 = np.array([dx, dy, 0.0])
    shp = p.shape
    return (((p.reshape(-1, 3) - piv3 - d3) @ Rz) + piv3).reshape(shp)


class MapFail(Exception):
    """A **declared** structural failure: no time map can be built from what the tree records (the synthesis
    candidates all fail their own gates · object columns / tail-trim contract) — the clip is quarantined (FAIL_MAP)."""


#: Row status heads — the producer owns the kind (the gate's one rule): FAIL_* = a declared data property (MapFail ·
#: FAIL_ASSIGN — canon · tree · resolver disagree on the object) → fails.txt (stage2 → cd_excluded → GUARD_FAIL) ·
#: ERROR_* = an input deficit or an undeclared exception (no base tree · no canon · stale canon fingerprint · a layout
#: the reader refuses · anything else) → errors.txt (stage2 → cd_error → BAKE_ERROR · systematic → the chunk stops,
#: ledger_states.error_systematic). Everything else is OK* / SKIP* / IDLE*.
OK_HEAD, FAIL_HEAD, ERROR_HEAD = "OK", "FAIL", "ERROR"
#: OK marks — 0 contact (V4 — a hand whose label is empty: not a failure, listed) · weak gate (G2 cannot discriminate
#: a resting object — not a failure, listed) · idle synthesis (a hand absent from the source given 0 contact). The OK
#: statuses of both ledgers are built only by ok_status and read only by is_ok · is_zero · is_weak (one vocabulary).
ZERO_TOKEN, WEAK_TOKEN, IDLE_SYNTH_TOKEN = "ZERO_CONTACT", "WEAK_GATE", "IDLE_SYNTH"


def ok_status(n_contact: int, *, weak: bool = False, idle_synth: bool = False) -> str:
    """An OK row status: OK · OK_ZERO_CONTACT · <either>_WEAK_GATE · OK_ZERO_CONTACT_IDLE_SYNTH."""
    marks = [m for m, on in ((ZERO_TOKEN, not n_contact), (WEAK_TOKEN, weak), (IDLE_SYNTH_TOKEN, idle_synth)) if on]
    return "_".join([OK_HEAD, *marks])


def is_fail(status: str) -> bool:
    return str(status).startswith(FAIL_HEAD)


def is_error(status: str) -> bool:
    return str(status).startswith(ERROR_HEAD)


def is_zero(status: str) -> bool:
    return str(status).startswith(f"{OK_HEAD}_{ZERO_TOKEN}")  # an OK mark — a FAIL message naming it is not


def is_weak(status: str) -> bool:
    return is_ok(status) and f"_{WEAK_TOKEN}" in str(status)


def is_ok(status: str) -> bool:
    return str(status).startswith(OK_HEAD)


def _tau_stored(tree: Path):
    """Read, as is, the time map the f0-first chain **stored directly** in the tree.

    `dexengine.placement.retime.warp` leaves `retime_tau` (cumulative time axis) · `retime_dt` ·
    `retime_s` (per-interval stretch) · `retime_src_idx` (final→previous frame map) in the final tree.
    This is the **warp's original record** and, for a v5 tree, the only source — the arm-only reproduction from the
    pre-retiming joints was deleted when retiming began to floor the stretch with the fingers (it could no longer
    reproduce the map). With the stored map, one tree is **self-contained**.

    Returns: (tau, T_pre) — tau is cumulative time (seconds) of length T_pre. Missing or off-contract → MapFail.
    """
    kt = tree / "0" / "trajectory_kinematic.npz"
    with np.load(kt) as z:
        if "retime_tau" not in z.files:
            raise MapFail("no stored time map (retime_tau)")
        tau = np.asarray(z["retime_tau"], dtype=np.float64)
        dt = float(z["retime_dt"]) if "retime_dt" in z.files else DT
    if tau.ndim != 1 or len(tau) < 2:
        raise MapFail(f"stored tau has a bad shape {tau.shape}")
    if abs(dt - DT) > 1e-12:
        raise MapFail(f"stored dt={dt} != {DT}")
    if not np.all(np.diff(tau) > 0):
        raise MapFail("stored tau is not monotonically increasing")
    return tau, len(tau)


def _stored_src_idx(tree: Path):
    """`retime_src_idx` as recorded in the tree (final frame → frame just before retiming). None if absent.

    **Not used as the source of the map.** On old clips this array is a post-hoc restoration, so
    `retime_map_resid` is not 0 (measured max 0.64 frames). It is purely a **watch** value,
    measuring "how far it diverges from the map we used".
    """
    kt = tree / "0" / "trajectory_kinematic.npz"
    with np.load(kt) as z:
        if "retime_src_idx" not in z.files:
            return None
        return np.asarray(z["retime_src_idx"], dtype=np.float64)


def _p_base_from(tau: np.ndarray, T_pre: int, T_fin: int, P: int, order: str, T_base: int) -> np.ndarray:
    """Tree frames 0..T_fin-1 → **fractional** positions in base (canon) frames."""
    if order == "f0_then_rt":
        # The pad is **inside** the retiming: the warp ran on (base + pad).
        p_pre = np.interp(np.arange(T_fin) * DT, tau, np.arange(T_pre, dtype=float))
        return np.clip(p_pre - P, 0.0, T_base - 1.0)
    if order == "rt_then_f0":
        # The pad was attached **outside** (in front of) the retiming: the first P frames copy source frame 0.
        out = np.zeros(T_fin, dtype=float)
        k = np.arange(P, T_fin)
        out[P:] = np.interp((k - P) * DT, tau, np.arange(T_pre, dtype=float))
        return np.clip(out, 0.0, T_base - 1.0)
    raise MapFail(f"unknown order: {order}")


def _obj_resid_mm(
    p_base: np.ndarray, qb: np.ndarray, qt: np.ndarray, hqd: int, place, ob_off: int, starts_t: list | None = None
) -> np.ndarray:
    """Object position residual (mm) — the **independent evidence** that the map is right.

    The object trajectory is resampled with the same time map as the contact and the placement is rigid, so
    sampling the canon-time-axis object through the map and applying the placement **must equal** the tree object.
    ob_off = object start column of the reference universe qpos (2·H_base; legacy sharpa 36, base layer 2·hqd = tree).
    starts_t = list of tree object start columns (qpos_layout.object_starts — in an articulated tree where a joint
               column sits between objects, 7-stride arithmetic reads the wrong column).
               None means legacy 7-stride.
    """
    cols = (
        list(starts_t)
        if starts_t
        else [2 * hqd + 7 * k for k in range(min((qb.shape[1] - ob_off) // 7, (qt.shape[1] - 2 * hqd) // 7))]
    )
    if not cols:
        raise MapFail(f"no object slot (base D={qb.shape[1]}, tree D={qt.shape[1]})")
    xs = np.arange(qb.shape[0], dtype=float)
    worst = np.zeros(len(p_base))
    for ct in cols:
        cb = ob_off + (ct - 2 * hqd)  # same object column in the reference universe (hand-block width only)
        ob = qb[:, cb : cb + 3]
        ot = qt[:, ct : ct + 3]
        pred = np.stack([np.interp(p_base, xs, ob[:, j]) for j in range(3)], axis=1)
        pred = _fwd(pred, *place)
        worst = np.maximum(worst, np.linalg.norm(ot - pred, axis=1) * 1e3)
    return worst


def _obj_resid_tree_mm(
    p_pre: np.ndarray, q_from: np.ndarray, q_to: np.ndarray, hqd: int, starts_t: list | None = None
) -> np.ndarray:
    """Object residual (mm) between tree placements — no placement transform, G2b (time map) only.
    Object columns from starts_t (qpos_layout), else legacy 7-stride."""
    cols = (
        list(starts_t)
        if starts_t
        else [2 * hqd + 7 * k for k in range((min(q_from.shape[1], q_to.shape[1]) - 2 * hqd) // 7)]
    )
    if not cols:
        raise MapFail(f"no object slot (from D={q_from.shape[1]}, to D={q_to.shape[1]})")
    xs = np.arange(q_from.shape[0], dtype=float)
    worst = np.zeros(len(p_pre))
    for c in cols:
        pred = np.stack([np.interp(p_pre, xs, q_from[:, c + jj]) for jj in range(3)], axis=1)
        worst = np.maximum(worst, np.linalg.norm(q_to[:, c : c + 3] - pred, axis=1) * 1e3)
    return worst


def _obj_pose(q: np.ndarray, base_col: int):
    """(R, p) — qpos object block (pos3 + quat wxyz4) → rotation matrix and position."""
    a = q[:, base_col : base_col + 7]
    return Rot.from_quat(a[:, [4, 5, 6, 3]]).as_matrix(), a[:, :3]


def _atomic_savez(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmpwrite.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def _read_ledger(ledger_path: Path, cols: list[str]) -> dict[tuple[str, str], list[str]]:
    """Rows of the existing ledger keyed by (clip, side); a ledger with another header is archived whole and left
    out (a new one is started)."""
    merged: dict[tuple[str, str], list[str]] = {}
    if ledger_path.exists():
        lines = ledger_path.read_text().splitlines()
        head = lines[0].split("\t") if lines else []
        if head != cols:
            ts = time.strftime("%Y%m%d_%H%M%S")
            arc = ledger_path.parent / "archive"
            arc.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ledger_path, arc / f"{ledger_path.name}.{ts}.schema.bak")
            print(
                f"[contact_ledger] ledger schema change detected — the old ledger is kept as "
                f"{data_rel(arc / f'{ledger_path.name}.{ts}.schema.bak')} "
                f"and a new one started (old {len(head)} columns → new {len(cols)} columns)",
                flush=True,
            )
            lines = []
        for ln in lines[1:]:
            f = ln.split("\t")
            if len(f) >= 2:
                merged[(f[0], f[1])] = f
    return merged


def _write_lists(all_rows: list[dict], lists: tuple, out_dir: Path, mode: str | None) -> None:
    """Each list file = ``clip side <column>`` of the rows its predicate accepts; replaced atomically too, so readers
    never see an empty list for a moment."""
    for name, pred, col in lists:
        rs = [r for r in all_rows if pred(r.get("status", ""))]
        dst = mode_record(out_dir / name, mode)
        ltmp = dst.parent / (dst.name + ".tmp")
        ltmp.write_text(("\n".join(f"{r['clip']}\t{r['side']}\t{r.get(col)}" for r in rs) + "\n") if rs else "")
        os.replace(ltmp, dst)


def merge_ledger(
    ledger_path: Path, rows: list[dict], cols: list[str], out_dir: Path, *, mode: str | None, lists: tuple
) -> list[dict]:
    """The one writer of a contact ledger (contact_derive ledger.tsv · contact_mano_base LEDGER.tsv): merge
    ``rows`` into the ledger on the (clip, side) key and re-derive every list from the whole merged ledger, **inside a
    single file lock**. Returns every merged row (dicts over ``cols``).

    ``ledger_path`` is the write run's ledger. ``mode`` = None for the write run (--apply); a run that writes nothing
    names its mode and every record of it — the ledger and all its lists — goes to the mode-tagged name
    (paths.mode_record — ``ledger.dryrun.tsv`` · ``fails.dryrun.txt``): the write ledger is the entry's layer-root
    evidence and a DRY run must never merge its rows into it. No default: the caller states the mode.
    ``lists`` = ((file name, status predicate — is_fail · is_error · is_zero, the row's second column), …) — the
    caller's vocabulary of this module's status heads; each file = ``clip side <column>`` per matching row.

    A full run drives several shards at once in CPU parallel. Overwriting the lists **wholesale, using only the rows
    this process handled** would let the shard that finishes later erase the earlier shard's lists. So they are
    re-extracted from the whole merged ledger and emitted inside the same lock section as the ledger write.
    A ledger whose header is not ``cols`` is archived whole and a new one started — appending old rows as is would
    shift the columns by one (status disappears and the OK count silently becomes 0).
    """
    ledger_path = mode_record(ledger_path, mode)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.parent / (ledger_path.name + ".lock")
    with open(lock_path, "a+") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            merged = _read_ledger(ledger_path, cols)
            for r in rows:
                merged[(r["clip"], r["side"])] = [str(r.get(c, "")) for c in cols]
            tmp = ledger_path.parent / (ledger_path.name + ".tmp")
            tmp.write_text("\t".join(cols) + "\n" + "\n".join("\t".join(v) for _, v in sorted(merged.items())) + "\n")
            os.replace(tmp, ledger_path)

            all_rows = [dict(zip(cols, v, strict=False)) for v in merged.values()]
            _write_lists(all_rows, lists, out_dir, mode)
            return all_rows
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


#: contact_derive's lists — fails.txt = FAIL_* rows · errors.txt = ERROR_* rows (FAIL_HEAD · ERROR_HEAD) · 0-contact.
DERIVE_LISTS = (
    ("fails.txt", is_fail, "status"),
    ("errors.txt", is_error, "status"),
    ("zero_contact.txt", is_zero, "T_fin"),
)


def finalize_outputs(ledger_path: Path, rows: list[dict], out_dir: Path, *, mode: str | None):
    """contact_derive's ledger (LEDGER_COLS · DERIVE_LISTS) through merge_ledger. Returns (cumulative OK count,
    cumulative fail rows, cumulative error rows, cumulative 0-contact rows, cumulative weak-gate rows)."""
    all_rows = merge_ledger(ledger_path, rows, LEDGER_COLS, out_dir, mode=mode, lists=DERIVE_LISTS)
    st = [str(r.get("status", "")) for r in all_rows]

    def pick(pred) -> list[dict]:
        return [r for r, s in zip(all_rows, st, strict=True) if pred(s)]

    return (sum(map(is_ok, st)), pick(is_fail), pick(is_error), pick(is_zero), pick(is_weak))
