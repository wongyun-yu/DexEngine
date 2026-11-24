"""Dense L1 set render: the member list of dense_traj (variant trajectories) → the canonical per-clip render
(record render clip.stage2_clip, orig round, one look-D episode per variant by default, one session) → one store call.

The variant trees live in the shadow root of the physics step, so the data root is that shadow and cell.seq_dir =
<shadow>/<seq_rel>. No new rule — the canonical render and store functions are called as they are. Writes only <out>.
Markers [DENSE_RENDER] …, last line `[DENSE_RENDER] DONE …`.

  python -m experiments.policies.dense_render --members <members.json> --out <dir> --gpu <id> --main <data root>
      [--tree <code tree>] [--looks-all [--look-rows 0,1 [--obj-colors red,… [--color-keep]] | --print-looks]]

Look options (default: look D only):
  --looks-all        look D + the random looks (canonical per_look, seed rule unchanged).
  --look-rows R,…    only the random looks of those deck rows (no look D; deck, key and sampler unchanged — deck row 0
  =
                     first-layer looks, rows 1–3 = second-layer looks). Needs --looks-all; rows outside the deck or
                     repeated die.
  --obj-colors C,…   (one look row) cell j gets an episode painted with colour j (augvis catalog OBJ_COLORS; look =
                     <cell>_<colour>, look_base = cell; room, light and distractors as the cell). Marker [OBJ_COLOR] on.
  --color-keep       with --obj-colors: augvis catalog color_keep objects keep their texture. Marker [COLOR_KEEP] on.
  --print-looks      (rows 0,1,2,3) every deck cell gets one print look (augvis catalog print_look — green, blue or the
                     original texture; seeded by variant and look). Marker [PRINT_LOOKS] on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: Switch markers and DONE fields that the render job's gate greps in this log.
OBJ_COLOR_ON = "[OBJ_COLOR] on"
COLOR_KEEP_ON = "[COLOR_KEEP] on"
PRINT_LOOKS_ON = "[PRINT_LOOKS] on"


def done_line(episodes: int, frames: int, skipped: int, render_s, store_s: float, ds) -> str:
    """Body of the last line `[DENSE_RENDER] DONE episodes <n> · … · skipped <n> · …`."""
    return (
        f"DONE episodes {episodes} · frames {frames} · skipped {skipped} · render s {render_s} · store "
        f"{store_s:.0f} s · {ds}"
    )


def say(*x) -> None:
    print("[DENSE_RENDER]", *x, flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--main", type=Path, required=True, help="base data root (the cell's checkpoint and judge env)")
    ap.add_argument("--tree", type=Path, default=ROOT, help="code tree the render runs from")
    ap.add_argument("--looks-all", action="store_true")
    ap.add_argument("--look-rows", default="")
    ap.add_argument("--obj-colors", default="")
    ap.add_argument("--color-keep", action="store_true")
    ap.add_argument("--print-looks", action="store_true")
    return ap.parse_args(argv)


def _install_look_rows(a, EPI, rows):
    """--look-rows: episodes = the given deck rows of the orig round, RANDOM_PER_TRAJ looks each (needs --looks-all)."""
    if a.look_rows:
        from dexengine.record.augment import policy as AP

        n_rows = EPI.deck_args(AP.PER_CLIP)["n_rows"]
        rows = [int(x) for x in a.look_rows.split(",")]
        if not a.looks_all or len(set(rows)) != len(rows) or any(r < 0 or r >= n_rows for r in rows):
            raise SystemExit(
                f"[DENSE_RENDER] --look-rows {a.look_rows!r} — needs --looks-all · rows 0–{n_rows - 1} · no repeats"
            )

        def episodes_rows(rnd, orig, augs, n_rows, row0=None):
            if rnd != "orig" or augs or row0 is not None:
                raise SystemExit(
                    f"[DENSE_RENDER] --look-rows takes the orig round and one trajectory — {rnd} {augs} {row0}"
                )
            R = EPI.RANDOM_PER_TRAJ
            return [
                {"traj": orig, "look": EPI.look_name(r, R * r + j, n_rows), "row": r, "col": R * r + j}
                for r in rows
                for j in range(R)
            ]

        EPI.episodes = episodes_rows
        say("[LOOK_ROWS]", rows, "· no look D · deck unchanged · looks per variant", EPI.RANDOM_PER_TRAJ * len(rows))
    return rows


def _install_obj_colors(a, EPI, rows, cols):
    """--obj-colors: episodes = one look row painted in each given object colour (optional --color-keep)."""
    if cols:
        from dexengine.record.looks.augvis import catalog as AC

        if (
            not a.look_rows
            or len(rows) != 1
            or len(cols) > EPI.RANDOM_PER_TRAJ
            or len(set(cols)) != len(cols)
            or any(c not in AC.OBJ_COLORS for c in cols)
        ):
            raise SystemExit(
                f"[DENSE_RENDER] --obj-colors {a.obj_colors!r} — one --look-rows row · colours ≤ "
                f"{EPI.RANDOM_PER_TRAJ} · from {sorted(AC.OBJ_COLORS)}"
            )
        if a.color_keep:
            say(COLOR_KEEP_ON, "· clip", AC.COLOR_KEEP_CLIPS, "· object ids = PRINT_KEEP")

        def episodes_colors(rnd, orig, augs, n_rows, row0=None):
            if rnd != "orig" or augs or row0 is not None:
                raise SystemExit(
                    f"[DENSE_RENDER] --obj-colors takes the orig round and one trajectory — {rnd} {augs} {row0}"
                )
            R, r = EPI.RANDOM_PER_TRAJ, rows[0]
            keep = AC.color_keep(orig) if a.color_keep else []
            return [
                dict(
                    traj=orig,
                    look=f"{EPI.look_name(r, R * r + j, n_rows)}_{c}",
                    look_base=EPI.look_name(r, R * r + j, n_rows),
                    row=r,
                    col=R * r + j,
                    obj_color=list(AC.OBJ_COLORS[c]),
                    **({"obj_keep": keep} if keep else {}),
                )
                for j, c in enumerate(cols)
            ]

        EPI.episodes = episodes_colors
        say(OBJ_COLOR_ON, cols, {c: AC.OBJ_COLORS[c] for c in cols}, "· row", rows[0], "· looks per variant", len(cols))
    elif a.color_keep:
        raise SystemExit("[DENSE_RENDER] --color-keep needs --obj-colors")


def _install_print_looks(a, EPI, rows, cols):
    """--print-looks: episodes = deck rows 0-3 with one print look kind per cell."""
    if a.print_looks:
        from dexengine.record.looks.augvis import catalog as AC

        if (
            cols
            or not a.look_rows
            or sorted(rows) != [0, 1, 2, 3]
            or len(AC.PRINT_LOOK_KINDS) != EPI.RANDOM_PER_TRAJ * 4
        ):
            raise SystemExit(
                f"[DENSE_RENDER] --print-looks — no --obj-colors · --look-rows 0,1,2,3 · "
                f"{EPI.RANDOM_PER_TRAJ * 4} cells (colours {a.obj_colors!r} rows {a.look_rows!r} table "
                f"{len(AC.PRINT_LOOK_KINDS)})"
            )

        def episodes_print(rnd, orig, augs, n_rows, row0=None):
            if rnd != "orig" or augs or row0 is not None:
                raise SystemExit(
                    f"[DENSE_RENDER] --print-looks takes the orig round and one trajectory — {rnd} {augs} {row0}"
                )
            R, eps = EPI.RANDOM_PER_TRAJ, []
            keep = AC.print_keep(orig)  # objects not painted (PRINT_KEEP)
            for r in rows:
                for j in range(R):
                    col, base = R * r + j, EPI.look_name(r, R * r + j, n_rows)
                    kind = AC.PRINT_LOOK_KINDS[col]
                    look = f"{base}_{AC.PRINT_LOOK_SUFFIX[kind]}"
                    eps.append(
                        dict(
                            traj=orig,
                            look=look,
                            look_base=base,
                            row=r,
                            col=col,
                            **AC.print_look(kind, f"{orig}|{look}"),
                            **({"obj_keep": keep} if keep else {}),
                        )
                    )
            return eps

        EPI.episodes = episodes_print
        say(
            PRINT_LOOKS_ON,
            {k: AC.PRINT_LOOK_KINDS.count(k) for k in AC.PRINT_LOOK_SUFFIX},
            "· noise",
            AC.PRINT_NOISE,
            "· looks per variant",
            len(AC.PRINT_LOOK_KINDS),
            "· kept",
            AC.PRINT_KEEP,
        )


def install_looks(a, EPI) -> list[int]:
    """Replace the episode list rule for --look-rows / --obj-colors / --print-looks (module docstring)."""
    rows: list[int] = []
    rows = _install_look_rows(a, EPI, rows)
    cols = [c for c in a.obj_colors.split(",") if c]
    _install_obj_colors(a, EPI, rows, cols)
    _install_print_looks(a, EPI, rows, cols)
    return rows


def store_root(sr: Path, dsn: str, main: Path) -> None:
    """Store data root <out>/storeroot: the store's K oracle (dexengine.record.store.slots.scan) writes a scan cache
    next to the statement (data/<ds>/collect/) of the data root. It must not land in the base checkout, so a registered
    shadow root is built (siblings link to the base, read only) and only data/<ds>/collect with its cache file is a real
    folder. Built before the shadow guard is armed (data root ≠ shadow) — handling base links after it would die."""
    from dexengine.core import paths as P
    from dexengine.record.augment import scene as AS

    assert P.data_layout._SHADOW_GUARD not in sys.modules, (
        "the shadow guard is already armed — cannot build the store root"
    )
    (sr / "data" / dsn / "collect").mkdir(parents=True, exist_ok=True)
    cp = sr / "data" / dsn / "collect" / f"stmt_scan_cache__rby1_wuji2__{dsn}.json"
    if not cp.exists():
        cp.write_text("{}")  # empty cache = counted again (miss); a real file, so no sibling link to the base cache
    AS.init_shadow(sr, main)
    AS.relink_shadow(sr, main)
    assert not cp.is_symlink() and (sr / "data" / dsn / "collect" / "stated_k__rby1_wuji2.json").is_symlink()


def _stage_ns(a, out, tree, shadow):
    """The stage-2 namespace of one dense render (shadow main root, local queue, orig round, no plan)."""
    ns = types.SimpleNamespace(
        hand="wuji2",
        main=shadow,
        out=out,
        tree=tree,
        gpu=str(a.gpu),
        queue_root=str(out / "queue"),
        round="orig",
        part="",
        orig_fov_out="",
        clips="",
        plan=False,
    )
    return ns


def _render_items(ns, cell, M, CCL):
    """Render every member variant (stage 2); returns its episodes, the skipped clips (reason) and seconds per clip."""
    eps, skip, sec = [], {}, {}
    for it in M["items"]:
        t0 = time.time()
        got, why = CCL.stage2_clip(ns, cell, it["clip"], [it], "orig")
        sec[it["clip"]] = round(time.time() - t0, 1)
        if why:
            skip[it["clip"]] = why
            say(f"{it['clip']} skipped {why} · {sec[it['clip']]} s")
            continue
        eps += got
        say(f"{it['clip']} episodes {len(got)} · {sec[it['clip']]} s")
    return eps, skip, sec


def _looks_label(a, EPI):
    """The looks label recorded in the source meta (look D, all, or the rows/colours/print selection)."""
    looks = (
        (
            (
                f"rows:{a.look_rows}"
                + (f"|colors:{a.obj_colors}" if a.obj_colors else "")
                + ("|print" if a.print_looks else "")
            )
            if a.look_rows
            else "all"
        )
        if a.looks_all
        else EPI.LOOK_D
    )
    return looks


def _store_source(M, cell, mj, shadow, tree, switches, rnd, skip, sec, looks, C, CM, CRN):
    """The source record stored with the episodes (cell, switches, round, code fingerprint, loader, dense rule)."""
    src = {
        "hand": "wuji2",
        "arm": "rby1",
        "entry": "dense_l1_render",
        "cells": [
            {
                "campaign": M["campaign"],
                "bundle": M["bundle"],
                "ckpt": str(cell["ckpt"]),
                "ckpt_sha256": cell["ckpt_sha"],
                "iter": cell["iter"],
                "label": cell["label"],
            }
        ],
        "switches": switches,
        "round": "orig",
        "round_switch": rnd,
        "robot_type": "rby1_wuji2",
        "code_fp": C.code_fingerprint(tree),
        "tree": str(tree),
        "controller": "PD joint position control — action.joint_target = absolute target (controller setpoint) · "
        "action = policy output",
        "loader": CM.loader_meta(CRN.FPS),
        "round_excluded": skip,
        "dense": {
            "rule": "region_near_canon_rank",
            "source_clip": M["clip"],
            "members": str(mj),
            "shadow": str(shadow),
            "looks": looks,
            "render_sec": sec,
        },
    }
    return src


def main(argv=None) -> int:
    a = parse(argv)
    out, mj, tree, base = a.out.resolve(), a.members, a.tree.resolve(), a.main.resolve()
    if a.looks_all:
        os.environ.pop("DEXENGINE_COLLECT_LOOKS", None)
    else:
        os.environ["DEXENGINE_COLLECT_LOOKS"] = "D"
    M = json.loads(mj.read_text())
    shadow = Path(M["shadow"]).resolve()
    os.chdir(tree)
    from dexengine.record import bundle as C
    from dexengine.record import bundle_cell as BC
    from dexengine.record import chain as CANON
    from dexengine.record import episodes as EPI
    from dexengine.record.render import clip as CCL
    from dexengine.record.render import episode as CRN
    from dexengine.record.store import meta as CM
    from dexengine.record.store import store as ST

    rnd = CANON.apply_round("orig")
    switches = CANON.switches()
    assert EPI.looks_mode() == ("" if a.looks_all else EPI.LOOK_D)
    if a.looks_all:
        say("[LOOKS] all — look D + random", EPI.RANDOM_PER_TRAJ, "· assets", os.environ.get("AUGVIS_ASSETS"))
    install_looks(a, EPI)
    ns = _stage_ns(a, out, tree, shadow)
    (out / "queue").mkdir(parents=True, exist_ok=True)
    os.environ.pop("DEXENGINE_MAIN_ROOT", None)  # the cell (checkpoint, judge env) resolves from the base data root
    dsn = M["seq_rel"].split("/")[1]
    sr = out / "storeroot"
    store_root(sr, dsn, base)
    cell = BC.cell_of(types.SimpleNamespace(hand="wuji2", main=base), M["campaign"], M["bundle"])
    os.environ["DEXENGINE_MAIN_ROOT"] = str(shadow)  # render and tree record = the shadow root (as the physics session)
    cell = dict(cell, seq_dir=str(shadow / M["seq_rel"]))
    say(
        f"{M['clip']} · variants {len(M['items'])} · shadow {shadow} · checkpoint {Path(str(cell['ckpt'])).name} · "
        f"GPU {a.gpu} · tree {tree}"
    )
    eps, skip, sec = _render_items(ns, cell, M, CCL)
    looks = _looks_label(a, EPI)
    src = _store_source(M, cell, mj, shadow, tree, switches, rnd, skip, sec, looks, C, CM, CRN)
    ds = out / "root" / "data" / dsn / "collect" / "rby1_wuji2"
    os.environ["DEXENGINE_MAIN_ROOT"] = str(sr)
    say(f"store data root {sr} (shadow · only data/{dsn}/collect is a real folder)")
    t0 = time.time()
    r = ST.store(ds, eps, src) if eps else {"episodes": 0, "frames": 0}
    say(done_line(r["episodes"], r["frames"], len(skip), sec, time.time() - t0, ds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
