"""Per-dataset search policy of the placement judge: robot-z candidates (dz layers) and the search origin.

dz layers (judge `--dz-cells`). dz = the robot base (root) z candidates, in 3.5 cm grid cells. The desk is fixed at
the world constant 0.76 m; the adopted dz is stamped into the arm IK npz as `base_z_off`, and the env places that
clip's robot root z from it. So this is not desk-height variation — it is a per-clip choice of robot height (on real
hardware, a height the torso joints would realise). TACO keeps the hand ~10 cm above the desk, so one layer (-5.5 cells
= -19 cm) is enough there; other datasets need their own layers. A dataset absent from DZ_CELLS raises (no silent
default). The judge compares its `--dz-cells` with this table and prints `[DZ_CELLS] dataset=… cells=…`.

Search origin (judge `--center-scene`). The judge searches placements on a scene-xy window of ±half_cells_x ×
±half_cells_y cells around an origin.
  off  (dataset absent from CENTER_SCENE — the default) the origin is the clip's own pivot in the capture frame
       (window displacement 0).
  on   every clip is first translated so that its pivot lands on the dataset's target point (scene x, y in metres)
       and the same window is searched around that point: d0(scene) = target − pivot(scene) (Judge._origin_shift).
  pivot = mean wrist position over the clip's active arms and strided frames, z dropped, in the judge's capture frame;
  scene = Rz(−90°)·capture (scene pivot = (cy, −cx)).
A dataset whose world origin sits far from the robot cannot be placed at all without a target (the window reaches
0.42 m). Both sides of the table are checked against the judged-dataset roster (DZ_CELLS): a key or a question outside
it dies, so a typo cannot pass silently as off. The judge input record carries the target only when it is on.
An explicit judge `--center-scene` that differs from the table dies unless OVERRIDE_ENV=1 and the run is isolated
(env layout.SANDBOX_ENV set).

No third-party dependencies (shell, judge and planners import it).
"""

from __future__ import annotations

import math
import os

from dexengine.core.errors import InputMissingError

OWNER = "dexengine/placement/search_policy.py"

# ── dz layers ─────────────────────────────────────────────────────────────────────────────────────────────────────
TACO_CANON = "-5.5"  # the TACO layer
_ONE = TACO_CANON
_THREE = "-7.5,-5.5,-3.5"  # -26 / -19 / -12 cm: hand 0.1-0.2 m above the desk, wide hand spread
_WIDE = "-5.5,-3.5,-1.5,0.5,2.5,4.5,6.5"  # -19 … +23 cm (7 cm apart): hot3d, lifting while standing. The +6.5
#   cell cap (desk at 0.53 m) avoids the lower-body box (height ~0.5 m)
#   hitting the desk edge — going lower needs a body/desk guard first

# Deskless scene (aerial): robot root z candidates -5.5 … +16.5 cells, 2 cells apart (= -19 … +58 cm, 12 layers). With
# no desk the +6.5 body/desk-edge cap disappears. In aerial scene mode this list replaces the dataset table.
AERIAL = ",".join(f"{-5.5 + 2 * k:.1f}" for k in range(12))
DZ_CELLS = {
    "taco": _ONE,
    "oakink2": _ONE,
    "hocap": _ONE,
    "dexcanvas": _ONE,
    "hrdexdb": _ONE,
    "arctic": _ONE,
    "grab": _THREE,
    "h2o": _THREE,
    "hot3d": _WIDE,
}
CELL_M = 0.035  # one dz cell (judge --dz-cells grid)
SCENES = ("table", "aerial")


def dz_cells_for(dataset: str, scene: str = "table") -> str:
    if scene == "aerial":
        return AERIAL
    if scene != "table":
        raise SystemExit(f"[dz_policy] scene mode '{scene}' must be table|aerial")
    try:
        return DZ_CELLS[dataset]
    except KeyError as e:
        raise InputMissingError(
            f"[dz_policy] no dz layer policy for dataset '{dataset}' — add it to {OWNER} DZ_CELLS (no silent default)"
        ) from e


def cells_to_cm(cells: str, cell_m: float = CELL_M) -> list[float]:
    return [round(float(c) * cell_m * 100, 1) for c in cells.split(",")]


def dz_bound_m() -> float:
    """The largest |robot root z offset| any layer list here can stamp (dataset table ∪ AERIAL) — the RL loader's
    sanity bound on `base_z_off` reads this, so a wider layer list widens it."""
    return max(abs(float(c)) for cells in (*DZ_CELLS.values(), AERIAL) for c in cells.split(",")) * CELL_M


# ── search origin ─────────────────────────────────────────────────────────────────────────────────────────────────
#: Explicit test override of the table for one judge run (`--center-scene=<x>,<y>` + this env = 1), allowed only in an
#: isolated run (`resolve`). The marker's source says so; the input record carries the value.
OVERRIDE_ENV = "V5_CENTER_SCENE_OVERRIDE"

#: dataset → target point (scene x, y, metres) where every clip's pivot is moved before the window search.
#: Absent = off. One key per dataset: every scene mode and every embodiment of that dataset reads the same row.
CENTER_SCENE: dict[str, tuple[float, float]] = {
    "hrdexdb": (-0.33, -0.15),
    "hot3d": (-0.34, -0.02),
    "h2o": (-0.34, -0.02),
    "grab": (-0.34, -0.02),
}


def norm(text) -> str:
    """Target text `x,y` → canonical text (`%.10g,%.10g` — '-0.18,0.0' and '-0.18,0' are the same value);
    '' / None → '' (off). Anything else dies (no silent off)."""
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    parts = s.split(",")
    try:
        v = [float(p) for p in parts]
    except ValueError:
        v = []
    if len(v) != 2 or not all(math.isfinite(x) for x in v):
        raise SystemExit(
            f"[CENTER_SCENE] target {text!r} is not a scene point 'x,y' (two finite reals) — owner {OWNER}"
        )
    return ",".join(format(x, ".10g") for x in v)


def _check_table() -> None:
    bad = sorted(set(CENTER_SCENE) - set(DZ_CELLS))
    if bad:
        raise SystemExit(
            f"[CENTER_SCENE] table datasets {bad} are not judged datasets (DZ_CELLS) — a typo would "
            f"switch nothing on and pass silently. Fix {OWNER}"
        )
    for ds, t in CENTER_SCENE.items():
        ok = (
            isinstance(t, (tuple, list))
            and len(t) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in t)
        )
        if not ok:
            raise SystemExit(
                f"[CENTER_SCENE] table row {ds}: {t!r} is not a scene point (x, y) (two finite reals). Fix {OWNER}"
            )


def target_for(dataset: str, scene: str = "table") -> str:
    """The target of this (dataset, scene) as canonical text, '' = off. One table for every scene mode.
    A dataset outside the judged roster dies — a typo or an axis name must not read as off."""
    if scene not in SCENES:
        raise SystemExit(f"[CENTER_SCENE] scene mode '{scene}' must be {'|'.join(SCENES)} — owner {OWNER}")
    _check_table()
    if dataset not in DZ_CELLS:
        raise SystemExit(
            f"[CENTER_SCENE] dataset {dataset!r} is not a judged dataset (DZ_CELLS) — a typo or an axis "
            f"name would pass silently as off (give the dataset name) · owner {OWNER}"
        )
    t = CENTER_SCENE.get(dataset)
    return "" if t is None else norm(f"{float(t[0])!r},{float(t[1])!r}")


def _sandbox(env) -> str:
    """Isolated-run root (env name owner layout.SANDBOX_ENV) · '' = a normal run."""
    from dexengine.core import layout as _L  # noqa: PLC0415 — the override path only

    return str(env.get(_L.SANDBOX_ENV, "") or "")


def resolve(dataset: str, scene: str, explicit, env=None) -> tuple[str, str]:
    """The effective target of one judge run → (canonical text | '' = off, source).
    explicit = the judge's `--center-scene` value ('' = not given → the table). Given and equal to the table → the
    table. Given and different → dies unless OVERRIDE_ENV=1 **and** an isolated run (layout.SANDBOX_ENV set), so an
    override can never be judged into the normal chunk / tree places."""
    env = os.environ if env is None else env
    pol = target_for(dataset, scene)
    ex = norm(explicit)
    if not ex or ex == pol:
        return pol, "policy"
    if env.get(OVERRIDE_ENV) != "1":
        raise SystemExit(
            f"[CENTER_SCENE] ds={dataset} explicit --center-scene={ex} ≠ policy {pol or 'off'} — for a "
            f"test set {OVERRIDE_ENV}=1 (the policy is {OWNER})"
        )
    sbx = _sandbox(env)
    if not sbx:
        raise SystemExit(
            f"[CENTER_SCENE] ds={dataset} override --center-scene={ex} runs only in an isolated run "
            f"(layout.SANDBOX_ENV) — written to the normal places it would mix into the training set"
        )
    return ex, f"test override ({OVERRIDE_ENV}=1 · isolated {sbx} · policy {pol or 'off'})"


def marker(dataset: str, scene: str, target: str, source: str) -> str:
    """The judge's applied-state line."""
    return (
        f"[CENTER_SCENE] ds={dataset} scene={scene} {'on' if target else 'off'} target={target or '-'} "
        f"source={source} (owner {OWNER})"
    )


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cells",
        nargs="+",
        metavar="<dataset> [<scene>]",
        help="print the dz layers of one dataset (scene default table) and exit",
    )
    ap.add_argument(
        "--target",
        nargs="+",
        metavar="<dataset> [<scene>]",
        help="print the search-origin target of one dataset (scene default table; empty line = off)",
    )
    ap.add_argument("datasets", nargs="*", metavar="<dataset>", help="datasets to tabulate (default all)")
    a = ap.parse_args(argv)
    for opt in ("cells", "target"):
        v = getattr(a, opt)
        if v is not None:
            if len(v) > 2 or a.datasets:
                ap.error(f"--{opt} <dataset> [<scene>] — at most two values, not with table arguments")
            fn = dz_cells_for if opt == "cells" else target_for
            print(fn(v[0], v[1] if len(v) == 2 else "table"))
            return 0
    _check_table()
    for ds in a.datasets or sorted(DZ_CELLS):
        print(f"{ds}\t{dz_cells_for(ds)}\t{cells_to_cm(dz_cells_for(ds))} cm\tcenter={target_for(ds) or 'off'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
