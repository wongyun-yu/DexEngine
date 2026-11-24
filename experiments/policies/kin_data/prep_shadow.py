"""E5 kinematic replay baseline (kin_data.prep): the shadow data root — its real layers, the input base copy,
the dataset axes and the run env shared by the source and variant branches."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from experiments.policies.kin_data.prep_tree import (
    ARM,
    HAND,
    HAND_DIR,
    MARK,
    SHADOW_PURPOSE,
    SIDES,
    inside,
)


def ds_flags(ds: str) -> dict:
    """Dataset launch axes — owner offsets.ds_flags (the rules table's launch flags)."""
    from experiments.policies.kin_data import offsets as KO  # noqa: PLC0415

    return KO.ds_flags(ds)


def setup_root(root: Path, main: Path, rows: list[dict], variants: bool = False) -> None:
    """Make the shadow root and the layers this run writes (base, trees, robots) real folders **first** — the shadow
    guard
    (data_layout.shadow_guard) arms the moment the data root is first resolved, and afterwards the unlink that turns a
    symlink into a real folder reads as a base write. The K statement location (data/<ds>/collect — kin layer
    k_statements) is always an empty real folder (render machines take only our statements). variants: the contact
    sidecar
    location of the variant base aliases (bake.alias_base; data_layout.sidecar_rel, resolved by the base registry) is
    made
    real too — that layer keeps its base sibling links (augment scene.shadow_dir; the alias points at its source
    sibling)."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.record.augment import scene as AS  # noqa: PLC0415

    root.mkdir(parents=True, exist_ok=True)
    P.data_layout.write_shadow_record(str(root), str(main), SHADOW_PURPOSE)
    for ch in main.iterdir():
        if ch.name == ".git":
            continue
        t = root / ch.name
        if not t.exists() and not t.is_symlink():
            AS._link(t, ch)
    rels, lrels = set(), set()
    for r in rows:
        combo = f"{ARM}_{HAND}{ds_flags(r['ds'])['variant']}"
        rels |= {
            f"data/{r['ds']}/base/{HAND_DIR}/{r['emb']}",
            f"data/{r['ds']}/trees/{combo}",
            f"precompute/{r['ds']}/robots/{combo}",
        }
        # K statement location (the folder of dexengine.record.store.slots.path) = an empty real folder — the kin
        # layer's k_statements rewrites it over the whole shadow tree
        # (with a base statement link, os.replace would follow the link into the base and die in the guard)
        rels.add(str(Path(P.data_layout.render("collect", r["ds"], combo=f"{ARM}_{HAND}")).parent))
        if variants:
            lrels |= {
                str(Path(P.data_layout.sidecar_rel(main, "contact_base", r["ds"], r["clip"], sd, HAND_DIR)).parent)
                for sd in SIDES
            }
    for rel in sorted(rels):
        if not (root / rel).is_dir() or not inside(root, root / rel):
            own_dir(root, main, rel)
    for rel in sorted(lrels):
        AS.shadow_dir(root, main, rel)
    print(f"{MARK} shadow {root} · base {main} · real layers {len(rels)} · base link layers {len(lrels)}", flush=True)


def own_dir(root: Path, main: Path, rel: str) -> Path:
    """Make root/rel an **empty** real folder (ancestors via shadow_dir, siblings = base symlinks) — no base links in
    this
    layer, so entries named like our outputs never mix in."""
    from dexengine.record.augment import scene as AS  # noqa: PLC0415

    rp = Path(rel)
    parent = AS.shadow_dir(root, main, str(rp.parent)) if str(rp.parent) != "." else root
    d = parent / rp.name
    if d.is_symlink():
        d.unlink()
    d.mkdir(parents=True, exist_ok=True)
    return d


def place_base(root: Path, main: Path, inputs: Path, ds: str, emb: str, clip: str) -> Path:
    """Hard-link the input copy (checked by sha) into the shadow base layer (same file system, no copy)."""
    rel = f"data/{ds}/base/{HAND_DIR}/{emb}"
    d = root / rel
    if not d.is_dir() or not inside(root, d):
        d = own_dir(root, main, rel)
    src, dst = inputs / rel / clip, d / clip
    if not (src / "task_info.json").is_file():
        raise SystemExit(f"{MARK} input base missing {src}")
    if not dst.exists():

        def _ln(s_, d_):
            try:
                os.link(s_, d_)
            except OSError:  # another file system (test root) — copy
                shutil.copy2(s_, d_)

        shutil.copytree(src, dst, copy_function=_ln)
    return dst


def set_env(ds: str, emb: str) -> dict:
    """Run env shared by the source and variant branches (dataset axes, idle-arm rule) — if they differ the idle-arm
    step dies.
    """
    fl = ds_flags(ds)
    os.environ["DEXENGINE_DATASET"], os.environ["DEXENGINE_EMBODIMENT"] = ds, emb
    os.environ["DEXENGINE_SCENE"] = fl["scene"] if fl["scene"] != "table" else ""
    os.environ["DEXENGINE_COMBO_VARIANT"] = fl["variant"]
    os.environ["DEXENGINE_ACTIVE_SIDES"] = "auto"
    return fl
