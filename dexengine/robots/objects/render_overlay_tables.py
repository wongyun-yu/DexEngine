"""The tables of an overlay bake (robots.objects.render_overlay --out): render_textures.tsv (one row per asset
link) and render_textures_table.json (mesh table · USD kind counts)."""

from __future__ import annotations

import collections
import json
import os

from dexengine.robots.objects import tex_packs as RTP


def img_cell(img) -> str:
    return "|".join(img) if isinstance(img, list) else (img or "-")


def write_link_table(a, packs, rows):
    """render_textures.tsv: one row per (asset, link); source columns only with --packs."""
    with open(os.path.join(a.out, "render_textures.tsv"), "w", encoding="utf-8") as fo:
        extra = packs is not None  # source columns only with --packs (off = table bytes unchanged)
        fo.write(
            "dataset\tasset\tartic\tlink\tstatus\tmesh_obj\timage\twhy"
            + ("".join(f"\t{c}" for c in RTP.TABLE_EXTRA) if extra else "")
            + "\n"
        )
        for r in rows:
            for link, x in r["links"].items():
                sr = x.get("source_row") or dict.fromkeys(RTP.TABLE_EXTRA, "-")
                fo.write(
                    f"{r['ds']}\t{r['oid']}\t{int(r['art'])}\t{link}\t{x['status']}\t{x.get('mesh_obj') or '-'}\t"
                    f"{img_cell(x.get('image'))}\t{x.get('why') or '-'}"
                    + ("".join(f"\t{sr[c]}" for c in RTP.TABLE_EXTRA) if extra else "")
                    + "\n"
                )


def write_usd_table(rows, a, table):
    """render_textures_table.json: mesh table and the USD kind counts per dataset."""
    usd_ct = collections.Counter(
        (
            r["ds"],
            "overlay(texture)"
            if any(x.get("st") is not None or x.get("color") is not None for x in r["links"].values())
            else ("no spec" if "*" in r["links"] else "overlay(no material)"),
        )
        for r in rows
    )
    with open(os.path.join(a.out, "render_textures_table.json"), "w", encoding="utf-8") as fj:
        json.dump(
            {"meshes": table, "usd": {f"{d}|{k}": v for (d, k), v in sorted(usd_ct.items())}},
            fj,
            ensure_ascii=False,
            indent=1,
        )
