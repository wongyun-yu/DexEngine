"""The `--what` answers about registries (retarget python, hand spec dirs, robot USD variants, runtime and combo
dirs) and run paths (run ids, bundles, cells, scope files) — branches ③ and ④ of dexengine.core.locations.cli."""

from __future__ import annotations

import sys

from dexengine.core.locations.hands import (
    ARM_IDS,
    HAND_IDS,
    RUNTIME_DIRS,
    combo_dirname,
    combo_ids,
    hand_spec_dir,
    mode_record,
    split_combo,
)
from dexengine.core.locations.modules import data_layout, load_registry
from dexengine.core.locations.roots import retarget_py, run_rel, work_area
from dexengine.core.locations.runs import hand_of_run_path, rl_bundle_dir, rl_cell_of, run_path


def _what_hand_spec_dir(a):
    """--what hand-spec-dir: the spec directory of --hand."""
    if not a.hand:
        print("[paths] --what hand-spec-dir needs --hand", file=sys.stderr)
        return 2
    try:
        print(hand_spec_dir(a.hand))
    except (KeyError, ValueError) as e:
        print(f"[paths] {e}", file=sys.stderr)
        return 2
    return 0


def _what_usd_variant(a):
    """--what plate-variant: the robot plate (USD variant) every process of (arm, hand) spawns."""
    # the variant is a value of (arm, hand) — through the hand's joint limit kind (HandSpec.sim_finger_limits)
    if not a.arm or not a.hand:
        print(f"[paths] --what {a.what} needs --arm · --hand (or DEXENGINE_ARM · DEXENGINE_HAND)", file=sys.stderr)
        return 2
    reg = load_registry("arm_specs")
    try:
        if a.arm not in reg.ARM_REGISTRY:
            raise KeyError(f"unregistered arm {a.arm!r} (dexengine.core.arms.ARM_REGISTRY)")
        v = reg.plate_variant(a.arm, a.hand)
        # This answer alone judges an empty value (the shell tables only trust the rc): an arm with a generation
        # axis always has a non-empty plate (the render body's material tail, plus the collider revision), so an
        # empty answer there is a broken table, never handed out. For an arm without a generation axis (a floating
        # hand — canon_variant '') the empty value is that arm's only plate.
        if not v and a.arm in reg.ARM_GENERATIONS:
            raise ValueError(
                f"arm {a.arm!r} (has a generation axis) · hand {a.hand!r}: {a.what} is empty — "
                "a plate of an arm with a generation axis carries the render material tail, so the table is "
                "broken (dexengine.core.arms.ARM_GENERATIONS)"
            )
        print(v)
    except (KeyError, ValueError) as e:
        print(f"[paths] {e}", file=sys.stderr)
        return 2
    return 0


def _what_arm_joint_pd(a):
    """--what arm-joint-pd: 1 when the arm has joint PD, 0 for a floating hand."""
    # 1 = yes · 0 = floating hand (ArmSpec.is_floating — no arm joints, wrist = wrench). Owner dexengine.core.arms.
    # One reader: the canonical recipe table (ARM_JOINT_PD) — the three launchers trust only that value.
    reg = load_registry("arm_specs")
    if a.arm not in reg.ARM_REGISTRY:
        print(
            f"[paths] --what arm-joint-pd: unregistered arm {a.arm!r} "
            "(dexengine.core.arms.ARM_REGISTRY — --arm or DEXENGINE_ARM)",
            file=sys.stderr,
        )
        return 2
    print(0 if reg.ARM_REGISTRY[a.arm].is_floating else 1)
    return 0


def _what_work_area(a):
    """--what work-area: a work area of this process's run tree."""
    if a.area not in data_layout.RUN_WORK_AREAS:  #   in the --rel form (run_rel): code-tree relative when the run
        print(
            f"[paths] --what work-area needs --area {'|'.join(data_layout.RUN_WORK_AREAS)} (got {a.area!r})",
            file=sys.stderr,
        )
        return 2
    print(run_rel(work_area(a.area)))  #   tree is this code tree (the shell cds there), else absolute
    return 0


def _what_split_combo(a):
    """--what split-combo: '<arm><TAB><hand><TAB><variant>' of a combo id."""
    try:
        arm, hand, var = split_combo(a.combo, ARM_IDS, HAND_IDS)
    except ValueError as e:
        print(f"[paths --what split-combo] {e}", file=sys.stderr)
        return 2
    print(f"{arm}\t{hand}\t{var}")
    return 0


def what_registry(a, ap) -> int | None:
    """`--what` branch ③: registry answers (retarget-py · hand-spec-dir(s) · USD variants (sim · render) ·
    runtime-dirs · run-ckpt-root · work-area · combo-dirs · combo-ids).
    rc when handled."""
    # Hand → spec directory. The exit that keeps **the shell from re-typing the table as a case
    # statement**.
    #   ROBOTS=$(python3 dexengine/core/paths.py --what hand-spec-dir --hand "$HAND") || exit 2
    if a.what == "retarget-py":  # owner of the retargeting python = retarget_py() (shell window)
        print(retarget_py())
        return 0
    if a.what == "hand-spec-dir":
        return _what_hand_spec_dir(a)
    if a.what == "plate-variant":  # the robot plate every process spawns — owner dexengine.core.arms.plate_variant
        return _what_usd_variant(a)
    if a.what == "arm-joint-pd":  # does the arm have arm joint PD (the consumer of --arm_kp/--arm_kd gains)?
        return _what_arm_joint_pd(a)
    if a.what == "run-ckpt-root":  # trainer checkpoint area's name (layout.RUN_CKPT_ROOT, run-tree
        print(data_layout.RUN_CKPT_ROOT)  #   relative — the place is --what work-area --area work_ckpt)
        return 0
    if a.what == "work-area":  # a work area's place in this process's run tree (paths.work_area)
        return _what_work_area(a)
    if a.what == "runtime-dirs":  # runtime output roots without a dataset axis (logs …,
        print(" ".join(RUNTIME_DIRS))  #   data-root relative) — the layer checker matches roots too
        return 0
    if a.what == "hand-spec-dirs":  # every registered hand (space separated)
        print(" ".join(hand_spec_dir(h) for h in HAND_IDS))
        return 0
    if a.what == "combo-dirs":  # combo id + the arm's canonical generation suffix = asset dir
        print(" ".join(combo_dirname(c) for c in combo_ids()))
        return 0
    if a.what == "combo-ids":  # every registered (arm, hand) pair = a combo
        print(" ".join(combo_ids()))
        return 0
    if a.what == "combo-arm-hands":  # the same pairs, one "<arm> <hand>" per line (shell loops)
        for arm, hand in sorted(load_registry("mount_specs").MOUNT_REGISTRY):
            print(arm, hand)
        return 0
    if a.what == "split-combo":  # combo (variant incl.) → "<arm>\t<hand>\t<variant>"
        return _what_split_combo(a)
    return None


def _what_run_path(a):
    """--what run-path: one path, or 'tag<TAB>path' per stdin line with --pairs-stdin."""
    if not a.layer:
        print("[paths] --what run-path needs --layer (run layers: --what run-layers)", file=sys.stderr)
        return 2
    if a.pairs_stdin:  # same run_path rule per tag (no copy) — a failed line leaves only that tag empty
        hand = "" if a.layer in data_layout.RUN_SHARED else a.hand
        for line in sys.stdin:
            tag, _, att = line.rstrip("\n").partition("\t")
            if not tag:
                continue
            path = ""
            if att and att != "-":
                try:
                    p = run_path(a.layer, hand, tag=tag, attempt=att)
                    path = str(run_rel(p) if a.rel else p)
                except (KeyError, ValueError) as e:
                    print(f"[paths] {tag}: {e}", file=sys.stderr)
            print(f"{tag}\t{path}")
        return 0
    axes = {
        k: v
        for k, v in (
            ("tag", a.tag),
            ("campaign", a.campaign),
            ("tool", a.tool),
            ("attempt", a.attempt),
            ("dataset", a.dataset),
            ("bundle", a.bundle),
            ("track", a.track),
        )
        if v
    }
    try:
        hand = "" if a.layer in data_layout.RUN_SHARED else a.hand
        p = run_path(a.layer, hand, **axes)
        # --rel: repo-relative (transfer lists and tars must be **repo-relative** to unpack in the same place
        # on another machine).
        print(run_rel(p) if a.rel else p)
    except (KeyError, ValueError) as e:
        print(f"[paths] {e}", file=sys.stderr)
        return 2
    return 0


def _what_rl_bundle_dir(a):
    """--what rl-bundle-dir: the RL bundle cell of the named datasets."""
    try:
        dl = [d for d in (a.dataset or "").split(",") if d]
        if len(set(dl)) != len(dl):
            raise ValueError(f"--dataset names a dataset twice: {dl} — not merged silently")
        ds = dict.fromkeys(dl, 1)
        print(rl_bundle_dir(a.hand, a.campaign, ds, a.bundle, root=a.tree or None))
    except (KeyError, ValueError) as e:
        print(f"[paths] {e}", file=sys.stderr)
        return 2
    return 0


def _what_rl_cell_of(a):
    """--what rl-cell-of: 'hand<TAB>layer' when the path is an RL bundle cell, else rc 1."""
    if not a.path:
        print("[paths] --what rl-cell-of needs --path", file=sys.stderr)
        return 2
    got = rl_cell_of(a.path, root=a.tree or None)
    if got is None:
        return 1
    print(f"{got[0]}\t{got[1]}")
    return 0


def _what_mode_record(a):
    """--what mode-record: the record slot of a run that writes nothing."""
    try:
        if not a.path:
            raise ValueError("--what mode-record needs --path (the record slot of the writing run)")
        print(mode_record(a.path, a.mode))
    except ValueError as e:
        print(f"[paths] {e}", file=sys.stderr)
        return 2
    return 0


def _what_hand_of_path(a):
    """--what hand-of-path: the hand root of a path (none: rc 1)."""
    if not a.path:
        print("[paths] --what hand-of-path needs --path", file=sys.stderr)
        return 2
    h = hand_of_run_path(a.path)
    if h is None:
        return 1
    print(h)
    return 0


def what_run(a, ap) -> int | None:
    """`--what` branch ④: run layers (run-path · run-layers · mode-record · hand-of-path). rc when handled, else
    None."""
    # ── Run layer answers (per-hand directories): the shell never joins `overfit/…`·`records/…` by hand ──
    #   CH=$(python3 dexengine/core/paths.py --what run-path --layer run_chunks --hand "$HAND") || exit 2
    #   python3 dexengine/core/paths.py --what run-path --layer run_campaign --hand <hand> --campaign <campaign>
    if a.what == "run-path":
        return _what_run_path(a)
    if a.what == "rl-bundle-dir":  # RL bundle cell (one dataset → its cell · several → mixed)
        return _what_rl_bundle_dir(a)
    if a.what == "run-layers":  # run layer table (layer · template · writer if shared)
        for k, v in data_layout.RUN_LAYERS.items():
            print(f"{k}\t{v}\thand")
        for k, (v, w) in data_layout.RUN_SHARED.items():
            print(f"{k}\t{v}\tshared:{w}")
        return 0
    if a.what == "rl-cell-of":  # "hand<TAB>layer" if the path is an RL bundle cell (else rc 1)
        return _what_rl_cell_of(a)
    if a.what == "mode-record":  # record slot of a no-write run (the rule is mode_record alone)
        return _what_mode_record(a)
    if a.what == "hand-of-path":  # hand root the path belongs to (none: rc 1, no output)
        return _what_hand_of_path(a)
    return None
