#!/usr/bin/env python3
"""E5 kinematic data — put contact sidecars (0/contact_mano_<side>.npz) into the kinematic trees. Sources = the
canonical dexengine.data.contact.derive (base → tree; kinematic trees have no retiming, so the time map is the identity
'none') called --apply --in-place per (combo, dataset, emb) group. Variants (<clip>_pvNN) = the source tree's sidecar
through the
canonical bake_post.remap_contacts (no retiming → identity time axis). Sources must finish first (in one run: source
groups → variants).

Why: the render and record entry (run_inputs.clip_inputs) reads the tree's contact sidecar (kit:contact), and the bake
  skips copying it in base-layer trees. No second contact path is made.
Shadow root: derive's ledger location (precompute/<ds>/contact_derive/<combo>) is made a real folder **before** the data
  root is resolved (prep_shadow.own_dir — else the shadow guard kills it as a base write). At the end the active sides'
  sidecars are counted per tree.
  DEXENGINE_MAIN_ROOT=<shadow root> python3 experiments/policies/kin_data/contact.py --trees <tsv: ds emb tree>
      --root <root> --main <base>
Markers [KIN_CONTACT] group … rc, last line [KIN_CONTACT] DONE trees n · sidecar present k · missing m
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # repo root
MARK = "[KIN_CONTACT]"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--py", default=sys.executable, help="python that runs derive (Isaac env — pinocchio · trimesh)")
    return ap.parse_args()


def _group_rows(rows: list[dict]) -> tuple[dict[tuple, list[str]], list[tuple]]:
    """Source trees grouped by (combo, dataset, emb) for derive, and the variant trees with their source clip."""
    from dexengine.core import paths as P0  # noqa: PLC0415 — owner of the variant names (variant_source)
    from experiments.policies.kin_data import (
        prep_shadow as KP,  # noqa: PLC0415 — owner of own_dir, ds_flags (and the arm/hand names it imports)
    )

    groups: dict[tuple, list[str]] = {}
    variants: list[
        tuple
    ] = []  # (combo, ds, emb, variant tree, source clip) — moved by the canonical bake_post.remap_contacts
    for r in rows:
        combo = f"{KP.ARM}_{KP.HAND}{KP.ds_flags(r['ds'])['variant']}"
        clip = P0.clip_name(r["tree"])
        if P0.variant_source(clip) != clip:
            variants.append((combo, r["ds"], r["emb"], r["tree"], P0.variant_source(clip)))
        else:
            groups.setdefault((combo, r["ds"], r["emb"]), []).append(r["tree"])
    return groups, variants


def _derive_groups(a: argparse.Namespace, groups: dict[tuple, list[str]], tmp: Path) -> int:
    """Run the canonical contact derive once per source group; returns the number of failed groups."""
    bad = 0
    for (combo, ds, emb), trees in sorted(groups.items()):
        lst = tmp / f"{combo}__{ds}__{emb}.txt"
        lst.write_text("\n".join(t.removesuffix("_pl_rt") for t in trees) + "\n")
        summ = tmp / f"{combo}__{ds}__{emb}.json"
        cmd = [
            a.py,
            str(ROOT / "dexengine/data/contact/derive.py"),
            "--combo",
            combo,
            "--dataset",
            ds,
            "--emb",
            emb,
            "--src-layer",
            "base",
            "--clips-file",
            str(lst),
            "--apply",
            "--in-place",
            "--overwrite",
            "--summary-json",
            str(summ),
        ]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (tmp / f"{combo}__{ds}__{emb}.log").write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-50000:])
        s = json.loads(summ.read_text()) if summ.is_file() else {}
        print(f"{MARK} group {combo}/{ds}/{emb} trees {len(trees)} · rc {r.returncode} · summary {s}", flush=True)
        bad += r.returncode != 0
    return bad


def _remap_variants(a: argparse.Namespace, variants: list[tuple], bad: int) -> int:
    """Move each variant's source contact sidecar onto the variant time axis; returns the updated failure count."""
    from dexengine.core import paths as P  # noqa: PLC0415 — tree location (data root = shadow)

    # variants: the source tree's contact sidecar on the variant time axis — the canonical aug owner
    # (bake_post.remap_contacts) as is
    # (base → tree derive reads the variant's placement as the source placement and dies on the object residual)
    from dexengine.record.augment import bake_post as ABP  # noqa: PLC0415
    from dexengine.record.augment import place_src as APS
    from experiments.policies.kin_data import (
        prep_shadow as KP,  # noqa: PLC0415 — owner of own_dir, ds_flags (and the arm/hand names it imports)
    )

    srcs: dict = {}
    for combo, ds, emb, vt, clip in variants:
        if "e5_spider" in json.loads((P.trees_dir(combo, ds, emb) / vt / "task_info.json").read_text()):
            # E5 SPIDER trees (spider_arm data_tree.contacts_from_orig already wrote contacts on the SPIDER time axis;
            # the retiming map exists on one side only, so remap_contacts cannot handle them) · kin trees lack this key
            # (unchanged)
            print(f"{MARK} [SPIDER] variant {vt}: contact is written by spider_e5 (skipped)", flush=True)
            continue
        try:
            if (ds, emb, clip) not in srcs:
                ti0 = json.loads((P.trees_dir(combo, ds, emb) / P.tree_name(clip) / "task_info.json").read_text())
                srcs[(ds, emb, clip)] = APS.Src(
                    a.root, KP.HAND, ds, emb, clip, arm=KP.ARM, aerial=str(ti0.get("scene") or "") == "aerial"
                )
            done = ABP.remap_contacts(srcs[(ds, emb, clip)], P.trees_dir(combo, ds, emb) / vt)
            print(f"{MARK} variant {vt} ← {clip} · {done}", flush=True)
        except (Exception, SystemExit) as e:  # noqa: BLE001 — a failure is counted and the next one runs (the last line says it)
            bad += 1
            print(f"{MARK} variant {vt} failed {type(e).__name__}: {e}"[:400], flush=True)
    return bad


def _count_sidecars(groups: dict[tuple, list[str]], variants: list[tuple]) -> tuple[int, int, list[str]]:
    """Count present and missing contact sidecars over every tree and active side."""
    from dexengine.core import paths as P  # noqa: PLC0415 — tree location (data root = shadow)

    groups_all = dict(groups)
    for combo, ds, emb, vt, _c in variants:
        groups_all.setdefault((combo, ds, emb), []).append(vt)
    have = miss = 0
    missing = []
    for (combo, ds, emb), trees in groups_all.items():
        for t in trees:
            td = P.trees_dir(combo, ds, emb) / t
            act = json.loads((td / "task_info.json").read_text()).get("e5_kin", {}).get("reach", {}).get("per_side", {})
            for sd in act:
                if (td / "0" / f"contact_mano_{sd}.npz").is_file():
                    have += 1
                else:
                    miss += 1
                    missing.append(f"{t}/{sd}")
    return have, miss, missing


def main() -> int:
    a = _parse_args()
    if os.environ.get("DEXENGINE_MAIN_ROOT") != str(a.root):
        raise SystemExit(f"{MARK} DEXENGINE_MAIN_ROOT must equal --root")
    from experiments.policies.kin_data import (
        prep_shadow as KP,  # noqa: PLC0415 — owner of own_dir, ds_flags (and the arm/hand names it imports)
    )

    with open(a.trees) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    groups, variants = _group_rows(rows)
    for combo, ds, _emb in groups:  # make the ledger location real first (before resolving the data root)
        KP.own_dir(a.root, a.main, f"precompute/{ds}/contact_derive/{combo}")
    tmp = a.root / "_kin_contact"
    tmp.mkdir(exist_ok=True)
    bad = _derive_groups(a, groups, tmp)
    bad = _remap_variants(a, variants, bad)
    have, miss, missing = _count_sidecars(groups, variants)
    print(
        f"{MARK} DONE trees {len(rows)} · sidecar present {have} · missing {miss} · group failures {bad}"
        + (f" · missing e.g. {missing[:3]}" if missing else ""),
        flush=True,
    )
    return 0 if (miss == 0 and bad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
