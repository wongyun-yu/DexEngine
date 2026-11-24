"""PRE-KIT: resolve the robot (arm, hand, USD variant) of a trainer run and check it is ready to spawn.

Sets DEXENGINE_COMBO_USD_VARIANT (from --usd), DEXENGINE_COMBO_VARIANT (from a combo --sequence_dir), DEXENGINE_ARM
and DEXENGINE_HAND; the import-time cfg factories of the task package read them after Kit starts.
"""

from __future__ import annotations

import os
import sys

from dexengine.core import paths as _pp
from dexengine.robots import selection as _rs


def resolve_robot(parser, args_cli) -> None:
    """Resolve --usd, adopt the (arm, hand) of a combo --sequence_dir, check readiness and record the selection."""
    # --arm/--hand are required by the parser, so both values are always set.
    _arm_sel = args_cli.arm
    _hand_sel = args_cli.hand

    # --usd picks the robot USD variant for train, eval and render alike; it wins over DEXENGINE_COMBO_USD_VARIANT.
    _usd_sel = (getattr(args_cli, "usd", None) or "").strip()
    if getattr(args_cli, "usd", None) is not None:
        if _usd_sel.lower() == "legacy":
            parser.error("--usd legacy (the old torso-tilted generation) is gone; name the variant to spawn")
        else:
            _usd_var = _usd_sel
            # A full asset name ('rby1_sharpa_wave_chest00') is reduced to its variant by the combo parser
            # (paths.split_combo): a prefix match would let a hand whose name is a prefix of another (xhand of
            # xhand_pro) swallow the other's asset. Another (arm, hand)'s asset is refused.
            _u_arm, _u_hand, _u_var = _pp.split_combo(
                _usd_var, _rs.ARM_REGISTRY.keys(), _rs.HAND_REGISTRY.keys(), unsplit_ok=True
            )  # a bare variant ('chest00') is valid too
            if _u_arm and (_u_arm, _u_hand) != (_arm_sel, _hand_sel):
                parser.error(
                    f"--usd {_usd_sel!r} is an asset of {_u_arm}·{_u_hand} — "
                    f"it differs from --arm {_arm_sel} --hand {_hand_sel}"
                )
            if _u_arm:
                _usd_var = _u_var
            if _usd_var and not _usd_var.startswith("_"):  # 'chest00'
                _usd_var = "_" + _usd_var
        os.environ["DEXENGINE_COMBO_USD_VARIANT"] = _usd_var
        print(
            f"[usd] --usd {_usd_sel!r} -> variant {_usd_var or '(empty suffix)'} "
            f"(asset {_arm_sel}_{_hand_sel}{_usd_var})"
        )

    _robot_from_combo_tree(parser, args_cli, _arm_sel, _hand_sel)


def _robot_from_combo_tree(parser, args_cli, _arm_sel, _hand_sel):
    """(arm, hand) decided by a --sequence_dir that is a combo tree (its caches name the combo)."""
    # A --sequence_dir that is a combo tree decides (arm, hand): its caches are per (arm, hand), so a run with
    # another pair would replay another robot's IK. Without explicit --arm/--hand the data's pair is adopted; an
    # explicit pair that disagrees with the data is refused. Suffix and combo assembly live in dexengine.core.paths.
    try:
        _sd_combo = _pp.combo_from_seq_dir(getattr(args_cli, "sequence_dir", "") or "")
    except ValueError as _e:  # a view whose entries do not point into one combo tree root: no identity, no run
        parser.error(f"--sequence_dir: cannot determine its identity (combo): {_e}")
    if _sd_combo:
        try:
            _c_arm, _c_hand, _c_var = _pp.split_combo(_sd_combo, _rs.ARM_REGISTRY.keys(), _rs.HAND_REGISTRY.keys())
        except ValueError as _e:
            parser.error(f"--sequence_dir is a combo tree ({_sd_combo}) but {_e}")
        # A floating hand reads the data world of an arm, so identity is compared with that arm
        # (floating.data_world_arm).
        _cur = f"{_rs.data_world_arm(_arm_sel)}_{_hand_sel}{os.environ.get('DEXENGINE_COMBO_VARIANT', '')}"
        if _cur != _sd_combo:
            _explicit = any(a == f"--{k}" or a.startswith(f"--{k}=") for a in sys.argv for k in ("arm", "hand"))
            if _explicit and os.environ.get("DEXENGINE_ALLOW_COMBO_MISMATCH", "0") != "1":
                parser.error(
                    f"--sequence_dir combo ({_sd_combo}) and --arm/--hand ({_cur}) disagree — "
                    f"combo caches are per (arm, hand); mixing them replays another robot's IK. "
                    f"Use --arm {_c_arm} --hand {_c_hand}, or if this is intended, "
                    f"approve it explicitly with DEXENGINE_ALLOW_COMBO_MISMATCH=1."
                )
            if not _explicit:
                args_cli.arm, args_cli.hand = _c_arm, _c_hand
                _arm_sel, _hand_sel = _c_arm, _c_hand
                if _c_var:
                    os.environ["DEXENGINE_COMBO_VARIANT"] = _c_var
                print(
                    f"[combo] --sequence_dir is a combo tree ({_sd_combo}) → "
                    f"--arm {_c_arm} --hand {_c_hand}"
                    + (f" DEXENGINE_COMBO_VARIANT={_c_var}" if _c_var else "")
                    + " adopted automatically. Pass the flags to state them explicitly.",
                    flush=True,
                )

    # Readiness: specs registered -> mount present -> combo USD on disk (the first missing piece is named).
    _usd_cache = _rs.USD_CACHE_DIR
    _ready, _why = _rs.combo_ready(_arm_sel, _hand_sel, _usd_cache)
    if not _ready:
        parser.error(
            f"--arm {_arm_sel} --hand {_hand_sel}: pair not ready — {_why}. "
            "Porting order: HandSpec/ArmSpec registered (smoke-gated), "
            "MOUNT_REGISTRY entry, combo URDF and USD (dexengine.robots)."
        )
    # Generation check before Kit boots. dexengine.rl.tasks.scene.robot (where the USD is fixed) enforces it again after
    # Kit is up; checking here fails in seconds instead of minutes and prints the spawned robot at the top of the log.
    try:
        _rs.assert_generation_coherent(_arm_sel, _hand_sel, _usd_cache)
    except RuntimeError as _e:
        parser.error(str(_e))

    # The robot selection for the import-time cfg factories (dexengine.robots.selection reads these).
    os.environ["DEXENGINE_ARM"] = args_cli.arm
    os.environ["DEXENGINE_HAND"] = args_cli.hand
