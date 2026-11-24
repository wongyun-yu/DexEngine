"""Launch-time checks of the data a bimanual run reads (no Isaac Lab): the SEQ_DIR identity and the dataset layers.

A SEQ_DIR is a combo tree root (or a view folder whose entries link into one). Its combo must be the launch's
(arm, hand) — or the data world of a floating hand — and its variant fixes DEXENGINE_COMBO_VARIANT. Every layer root
the training entry reads (trees, contact sidecars, precompute) must exist, and the SEQ_DIR must resolve to the
`trees` root of this dataset and embodiment (or to a named, frozen roster root the caller pins with
DEXENGINE_ROSTER).
"""

from __future__ import annotations

import os
from pathlib import Path

from dexengine.core import arms
from dexengine.core import paths as P
from dexengine.rl.launch.recipe import RecipeError


def seq_dir_identity(seq_dir: str, arm: str, hand: str) -> dict:
    """Combo, variant, view flag, resolved tree root and named roster of a SEQ_DIR (dies on a foreign robot)."""
    try:
        combo, is_view = P.seq_dir_identity(seq_dir)
        m_arm, m_hand, var = P.split_combo(combo, P.ARM_IDS, P.HAND_SPEC_DIR)
        spec = arms.ARM_REGISTRY.get(arm)
        # The floating hand reads the data world of another arm (its experiment module says which); an arm body
        # reads its own.
        world_arm = (
            P.load_registry("floating_hand").data_world_arm(arm) if spec is not None and spec.is_floating else arm
        )
        if P.combo_id(m_arm, m_hand) != P.combo_id(world_arm, hand):
            raise ValueError(f"combo {combo} is not the launch robot DEXENGINE_ARM={arm} DEXENGINE_HAND={hand}")
    except (KeyError, ValueError) as e:
        raise RecipeError(f"cannot identify SEQ_DIR {seq_dir}: {e}") from e
    root = Path(seq_dir).resolve()
    if is_view:
        root = next(
            Path(os.path.realpath(root / e)).parent
            for e in sorted(os.listdir(root))
            if (root / e / "0" / "trajectory_kinematic.npz").is_file()
        )
    return {
        "combo": combo,
        "variant": var,
        "view": is_view,
        "root": os.path.realpath(root),
        "roster": P.named_roster_of(root),
    }


def check_bimanual(seq_dir: str, arm: str, hand: str, dataset: str, emb: str, env: dict, code_root: Path) -> str:
    """Run the bimanual data checks; return the combo variant (DEXENGINE_COMBO_VARIANT) the run must carry."""
    ident = seq_dir_identity(os.path.realpath(seq_dir), arm, hand)
    combo, var = ident["combo"], ident["variant"]
    if "DEXENGINE_COMBO_VARIANT" in env:
        if env["DEXENGINE_COMBO_VARIANT"] != var:
            raise RecipeError(
                f"DEXENGINE_COMBO_VARIANT={env['DEXENGINE_COMBO_VARIANT']!r} differs from the SEQ_DIR "
                f"variant {var!r} (combo {combo}); leave it unset to take the SEQ_DIR's"
            )
    else:
        print(f"[launch] DEXENGINE_COMBO_VARIANT={var!r} taken from the SEQ_DIR combo {combo}", flush=True)
    try:
        rows = P.entry_layer_roots(dataset, combo, emb)
    except (KeyError, ValueError) as e:
        raise RecipeError(f"dataset layers of dataset={dataset} emb={emb} combo={combo}: {e}") from e
    trees, seen = "", ""
    for layer, p in rows:
        if not Path(p).is_dir():
            raise RecipeError(f"dataset layer root missing: {p} (dataset={dataset} emb={emb} combo={combo})")
        if layer == "trees":
            trees = os.path.realpath(p)
        seen += f" {layer}={p}"
    roster = env.get("DEXENGINE_ROSTER", "")
    pinned = bool(roster) and roster == ident["roster"]
    if not pinned and trees != ident["root"]:
        raise RecipeError(
            f"SEQ_DIR resolves to {ident['root']}, not the trees root {trees or 'none'} of "
            f"dataset={dataset} emb={emb} combo={combo} (another dataset, embodiment or data root)"
        )
    print(
        f"[TREE] launch dataset={dataset} emb={emb} combo={combo} view={int(ident['view'])} root={ident['root']} "
        f"code={code_root} seq_dir={seq_dir}{seen}" + (f" roster={roster}" if pinned else ""),
        flush=True,
    )
    return var
