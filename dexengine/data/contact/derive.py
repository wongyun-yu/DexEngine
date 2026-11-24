#!/usr/bin/env python3
"""Contact canon → combo tree **derivation** (time-map synthesis + AND rule).

What this does
    The **combo-independent canon** (layer `contact_base`:
    `precompute/taco/contact_mano_base/<clip>_{right,left}.npz`, on the base time axis
    with no retiming) is moved onto this combo tree's time axis.

      · Boolean channels (contact, contact_bin, penetration, palm_penetration)
          → **AND rule**: True only when **both bracketing frames** of the source
            position a tree frame points at are True. Interpolate→cast (True if ≠ 0) has
            OR semantics and **over-reports** contact (dexengine.placement.retime.warp comment). AND is conservative,
            so a wrong label misses contact.
      · Continuous channels (dist, effective_dist, palm_*, witness_*) → linear interpolation.
      · witness_* are **positions**, so the placement (rigid transform) is re-applied.
        Distances and bins are rigid-invariant, so they are untouched (same contract as bake).

How the time map is built — **① stored map → ② synthesis** (not back-solving)
    ① **stored** (highest priority). Use the time map the f0-first chain left in
       the final tree — `retime_tau`·`retime_dt`(·`retime_s`) — **verbatim**: the warp's own
       record, so nothing is reproduced and nothing depends on the sidecar
       `arm_ik_poses_*/<tree>.pre_rt.npz` — one tree is self-contained. Same G1·G2 gates apply.
       `retime_src_idx` is **not used as a source** — on old clips it is a post-hoc restoration,
       so `retime_map_resid`>0 (measured max 0.64 frames); the ledger `srcidx_dev_f` only watches it.
    ② With no stored map, the warp is reproduced exactly from the sources recorded in the tree:
      task_info: retime_v2 / retimed_from / retime_safety / retime_ik_suffix /
                 retime_sides / trimmed_tail_frames / tail_trim_frames / hand_qpos_dim
      F0_REPAIR.json: n_blend_effective (number of pad frames P prepended in front)
    The two kinds of chain differ in **order** — get the order wrong and everything shifts by P frames:
      v5 (retime_v2="none"): placement → f0 repair (+P) → IK → time-axis stretch
          → tree frame k: p_pre = interp(k·dt, tau, 0..T_pre-1),  p_base = p_pre − P
          tau = **the stored map only** (①). The arm-only reproduction from the pre-retiming joints is gone: retiming
          also floors the stretch with the fingers (HandSpec applied limit), which the arm joints cannot reproduce —
          a v5 tree without a stored map has no candidate (MapFail, quarantined).
      legacy (retime_v2="combo"/"inherit"): placement → time axis → f0 repair (+P)
          → k < P is pad (source frame 0), k ≥ P is interp((k−P)·dt, tau, 0..T_pre-1)
          neither warp can be reproduced any more (both implementations were deleted) — such a tree
          derives only from a stored map, else it fails loudly with the reason.
    Candidates are tried in order and verified by **length match + object residual**.

Self-verification (on failure, **nothing is written; the clip is quarantined in the fail list**)
    G1 length: the frame count the synthesized warp predicts == the tree's actual frame count.
    G2 map residual: the object trajectory is resampled with the same time map, so
       |tree object − placement_transform(canon-time-axis object[map])| **must be 0** (measured <1e-6 mm).
       This is the independent evidence that the map is right. Over the threshold → fail.
    G2b discriminative power: if deliberately shifting the map by 2 frames barely changes the
       residual (the object is at rest), G2 cannot discriminate → mark WEAK_GATE (no silent pass).
    G3 left/right assignment: the canon tool_mesh_id == the tree task_info `<side>_object_mesh_dir`
       == the side_objects resolver's (mesh, slot). All three sources must agree.
    G4 detect hands with 0 contact frames → mark ZERO_CONTACT and collect them in a list.

env knob (default OFF — the tree's existing bytes stay as they are)
    (none)              DRY. Compute, verify and ledger only. Nothing is written to the tree; the ledger and the
                        lists go to the mode-tagged names (ledger.dryrun.tsv · fails.dryrun.txt — paths.mode_record),
                        never into the write run's ledger (the entry's layer-root evidence).
    --apply             writes only a **new file**, `<tree>/0/contact_mano_<side>.derived.npz`.
    --apply --in-place  replaces the existing `contact_mano_<side>.npz` (no backup — it can be
                        re-derived from the canon; one log line per replacement).

Outputs
    precompute/taco/contact_derive/<combo>/ledger.tsv       ledger (one row per clip×side)
    precompute/taco/contact_derive/<combo>/fails.txt        failure quarantine (FAIL_* — declared structural)
    precompute/taco/contact_derive/<combo>/errors.txt       input deficit · undeclared exception (ERROR_*)
    precompute/taco/contact_derive/<combo>/zero_contact.txt hands with 0 contact

Usage ($RETARGET_PY = python -m dexengine.core.paths --what retarget-py)
    $RETARGET_PY dexengine/data/contact/derive.py --combo rby1_sharpa_wave --limit 3
    $RETARGET_PY dexengine/data/contact/derive.py --combo rby1_sharpa_wave --apply --in-place
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import (  # noqa: E402
    SRC_LAYERS,
    contact_derive_ledger,
    default_src_layer,
    mode_record,
    read_clip_list,
    tree_name,
)
from dexengine.data.contact import (
    derive_tree as DTREE,  # the run's roots live with their readers (set by _configure_roots)
)
from dexengine.data.contact.derive_tree import _configure_roots, process_tree
from dexengine.data.contact.derive_util import (
    finalize_outputs,
)

HELP = "Contact canon -> combo trees: derive (time-map synthesis + AND rule)."


from dexengine.data.contact.derive_steps import (  # noqa: E402,F401  (re-exported)
    _ARM_REG,
    _COLL,
    SIDES,
    _arm_of,
    _assign_from_layout,
    _candidates,
    _refuse,
    _report,
    _tau_inherit,
    arm_registry,
    collision_mod,
    derive_side,
)


def _parse_args() -> argparse.Namespace:
    """Command line: combo/dataset/emb axes, source layer, clip selection, apply mode and the summary JSON."""
    ap = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combo", required=True)
    ap.add_argument(
        "--dataset",
        required=True,
        help="dataset id (axis of layer paths and the reference universe); no default, the caller states the axis",
    )
    ap.add_argument(
        "--emb", required=True, help="embodiment: right|left|bimanual; no default, the caller states the axis"
    )
    ap.add_argument(
        "--src-layer",
        dest="src_layer",
        default="",
        choices=list(SRC_LAYERS) + [""],
        help="source layer (base only, the tree's own layer); empty = paths.default_src_layer",
    )
    ap.add_argument("--clip", default=None, help="single base clip")
    ap.add_argument("--clips", default=None, help="separated by ';'")
    ap.add_argument("--clips-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--apply", action="store_true", help="write into the trees (default OFF). Alone, only new .derived.npz files."
    )
    ap.add_argument(
        "--in-place", action="store_true", help="with --apply: replace the existing contact_mano_<side>.npz (no backup)"
    )
    ap.add_argument(
        "--summary-json",
        dest="summary_json",
        default="",
        help="write this run's counts as machine-readable JSON. The consumer (stage2_ground.sh _cd_ND) reads "
        "only this file, not the human progress lines. Atomic write (temp+rename)",
    )
    return ap.parse_args()


def _select_trees(a, root: Path) -> list[Path]:
    """Trees under root to derive: all, or the clips named by --clips-file / --clips / --clip, capped by --limit."""
    if a.clips_file:
        names = [
            ln.strip() for ln in Path(a.clips_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")
        ]
    elif a.clips:
        names = [c.strip() for c in a.clips.split(";") if c.strip()]
    elif a.clip:
        names = [a.clip]
    else:
        names = None
    trees = sorted(p for p in root.iterdir() if p.is_dir() and (p / "task_info.json").exists())
    if names is not None:
        names = read_clip_list(names, dataset=a.dataset, label="contact_derive_combo")
        want = {tree_name(n) for n in names}
        trees = [t for t in trees if t.name in want]
    if a.limit:
        trees = trees[: a.limit]
    return trees


def _derive_trees(trees: list[Path], a, apply: bool, in_place: bool) -> list[dict]:
    """process_tree over the trees with a progress line per tree (every 100th tree above 50 trees)."""
    rows: list[dict] = []
    for i, t in enumerate(trees, 1):
        r = process_tree(t, a.combo, apply, in_place and apply, a.overwrite)
        rows += r
        if len(trees) <= 50 or i % 100 == 0 or i == len(trees):
            print(
                f"[{i}/{len(trees)}] {t.name}: "
                + " | ".join(
                    f"{x['side']}={x.get('status')}"
                    f"({x.get('contact_frames_base', '-')}→"
                    f"{x.get('contact_frames_out', '-')}f)"
                    for x in r
                ),
                flush=True,
            )
    return rows


def main() -> int:
    a = _parse_args()
    a.src_layer = a.src_layer or default_src_layer(a.dataset)
    _configure_roots(a.combo, a.dataset, a.emb, a.src_layer)
    print(
        f"[contact_derive] dataset={a.dataset} emb={a.emb} src_layer={a.src_layer} base_ref={DTREE.TRAJ} "
        f"trees={DTREE.GEN_V2} "
        f"canon={DTREE.CANON_DIR} robots={DTREE.ROBOTS} out={DTREE.OUT_ROOT}",
        flush=True,
    )

    root = DTREE.GEN_V2
    if not root.is_dir():
        raise InputMissingError(f"combo tree root missing: {root}")
    trees = _select_trees(a, root)
    in_place = bool(a.in_place)
    apply = bool(a.apply)
    mode = (
        "DRY (no writes)"
        if not apply
        else ("APPLY+IN-PLACE (replaces canonical)" if in_place else "APPLY (new .derived.npz files only)")
    )
    print(f"[contact_derive] {a.combo} trees {len(trees)} · mode {mode}", flush=True)
    if in_place and not apply:
        print("[contact_derive] --in-place needs --apply — continuing as DRY", flush=True)

    t0 = time.time()
    rows = _derive_trees(trees, a, apply, in_place)

    out_dir = DTREE.OUT_ROOT
    # The name's owner = paths (it also reads the entry layer root evidence). A DRY run never merges rows into the
    # write run's ledger (mode suffix, paths.mode_record).
    mode = None if apply else "dryrun"
    ledger = contact_derive_ledger(a.combo, a.dataset)
    cum = finalize_outputs(ledger, rows, out_dir, mode=mode)
    return _report(a, rows, mode_record(ledger, mode), out_dir, mode, cum, t0)


if __name__ == "__main__":
    sys.exit(main())
