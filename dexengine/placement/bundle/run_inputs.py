#!/usr/bin/env python3
"""Campaign run inputs — the one statement of which files a campaign bundle reads.

A bundle is one training run (SIDE=both|right|left over its roster list). What it reads is stated per clip (clip ·
combo · hand · dataset · emb · side — independent of how clips are bundled) along the training entry's own code
path (dexengine/rl/cli.py --sequence_dir: roster · layout · preflight → Kit → env), through the owners that path
calls (train_ready_util/datasets ledger · qpos_layout · paths.ClipPaths · artifact_paths · robots.selection ·
split_artic · train_set). Per bundle it adds the roster ledger and the robot, whose switches (split articulation ·
USD variant · IK suffix) come from the canonical launch table (scripts/_rl_recipe.sh, sourced for the bundle's
path — canon_env).

Every item carries its READER (the consumer that reads it):
  entry:*   the entry (before Kit) touches it
  kit:*     read after Kit starts (the named consumer is the reason)
  verify:*  the campaign manifest check reads it
  launch:*  the launcher needs it before the entry (layer-root = a file carrying a layer root the GPU host requires
            — paths.layer_root_records)
  render:*  only `Stater.render`: a record render's plate, F0_REPAIR.json, object overlays and look D files; never in
            a bundle's set (a bundle's (n, digest) does not change with it)
A statement, not a measurement: nothing here checks that an input exists — except where the reader itself reads a
file only if present (qpos_layout's legacy scene.xml). What cannot be stated dies: a named-roster manifest, a clip
without a main object on an active side, a bundle whose tree root or ledger axis disagrees with its owner.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

from dexengine.core import paths as P
from dexengine.core import stdio
from dexengine.core.errors import InputMissingError
from dexengine.data import rules as RR
from dexengine.placement.train_set import select as ST

#: Item kinds (written into FILES.tsv and bundle.json counts — data values).
KINDS = ("roster", "tree", "base", "trajectory", "cache", "object", "robot", "layer", "look")
READERS = {
    "entry:roster": "train_ready_util.resolve_roster → datasets.ledger_file — opens the folded ledger (roster)",
    "entry:layout": "qpos_layout.read — opens the tree task_info.json (entry preflight · verify tree_facts/run_pools)",
    "entry:layout-legacy": "qpos_legacy.legacy_slots — the scene.xml (if present) of a legacy tree without slot keys",
    "entry:discovery": "--sequence_dir discovery (stat) → trajectory_loader opens it",
    "entry:preflight": "--sequence_dir preflight (stat) → consumers after Kit open it "
    "(caches trajectory_manager·events · object USD spawn · surface_geometry)",
    "entry:robot-gate": "robots.selection — combo USD (stat) · bake meta config.yaml (opened) → Kit spawns the USD "
    "(floating hand = per-side split pair — selection.required_robot_files)",
    "entry:robot-identity": "selection.assert_generation_coherent → derived_usd.verify — derived record · source USD · "
    "render plate material source USD (md5) · hand stage meshes opened by the arm collider "
    "digest (identity_inputs)",
    "kit:loader-discovery": "trajectory_loader — the `*/0/trajectory_ikrollout.npz` glob that discovers clips",
    "kit:body-positions": "trajectory_manager._load_body_positions",
    "kit:contact": "contact_aware — contact_mano_<side>.npz (that side's contact term · bimanual states both sides)",
    "kit:keypoints": "task rewards · training — keypoints_fps2.npy (the diagnostic term is always registered)",
    "kit:split": "split_artic.split_usd_paths — left/right articulation USDs of DEXENGINE_BI_SPLIT_ARTIC=1",
    "kit:support": "env setup _support_rows — USDs of the support objects stamped by the bake (support_objects)",
    "kit:extra": "env setup build_extra_slots — what the extra-slot rule reads when building the env: USDs that can "
    "become extra slots · object meshes its geometry opens (extra_slots.env_inputs — none without "
    "extras)",
    "kit:scene-rules": "scene rules (carry replay · touch gate · envelope collider) — cooked hull dumps",
    "kit:float-settle": "env setup build_float_settle — S-float-settle's contact readings (float_settle.inputs: base "
    "tree task_info · contact_base/contact_all sidecars · their fingerprint inputs · an articulated main's MANO)",
    "kit:wrist-table": "trajectory_manager — a one-hand run opens the wrist table of both sides: wrist_poses_right "
    "always (missing dies) · the _left sibling folder per clip when present (a gap dies)",
    "kit:desk": "scene desk — desk_visual.desk_cfg → look_assets.willow_usd: the Willow desk of every scene, read from "
    "the look mirror under the data root (8 files, sha-pinned · NVIDIA content, never in a tree — render_unify)",
    "verify:base": "train_set.target_object_file — base-layer task_info (datasets with obj_key=base)",
    "verify:shape": "train_set.shape_files — surface points of a bundle's object key members (shape family "
    "re-measure · obj_key=base · articulated member: usd_artic)",
    "verify:sources": "campaign manifest sources — every folded ledger of the hand (axes without bundles included)",
    "launch:layer-root": "bimanual dataset axis — layer roots exist (paths.ENTRY_LAYER_ROOTS · layer_root_records)",
    "render:plate": "render — plate (robot_inputs under arms.plate_variant: combo USD · meta) · Stater.render",
    "render:f0-meta": "render — tree F0_REPAIR.json (opened if present · pad prepend) · Stater.render",
    "render:object": "render — overlay of each spawned object USD (artifact_paths.render_spawn_usds) · Stater.render",
    "render:look": "render — look D's files and what they open (look_assets.render_files) · Stater.render",
    "render:render-only": "render — render-only objects' index · record · anchor frames · surface samples "
    "(render_only.render_inputs)",
}
#: Readers only `Stater.render` states — never in a bundle's set.
RENDER_READERS = ("render:plate", "render:f0-meta", "render:object", "render:look", "render:render-only")
RENDER_SPAWNED = ("entry:preflight", "kit:extra")  # readers of the objects a render spawns as overlays (not supports)
SIDES = ("both", "right", "left")
#: Canonical launch-table values the robot items depend on (names in scripts/_rl_recipe.sh).
CANON_SWITCHES = ("DEXENGINE_BI_SPLIT_ARTIC", "DEXENGINE_COMBO_USD_VARIANT", "DEXENGINE_ARM_IK_DIR_SUFFIX")


class Item(NamedTuple):
    path: Path
    kind: str
    reader: str


def launch_env(b: dict, man: dict) -> list[str]:
    """The launch env of a bundle ("K=V"; GPU comes from the gate at launch). MI and every other recipe value come from
    the RL launcher's canonical recipe table (dexengine.rl.launch.recipe) — not restated here."""
    env = [f"DEXENGINE_ARM={man['arm']}", f"DEXENGINE_HAND={b['hand']}", f"DEXENGINE_DATASET={b['dataset']}"]
    env += [f"DEXENGINE_EMBODIMENT={b['emb']}"]
    env += [f"DEXENGINE_COMBO_VARIANT={b['variant']}"] if b["variant"] else []
    return env + [f"SIDE={b['side']}", f"SEQ_DIR=$PWD/{b['tree_root']}"]


#: table_values memo keyed by (table path · mtime_ns · names · the whole env): a changed table or env reads again.
_TV_MEMO: dict = {}


def _tv_key(names: tuple[str, ...], env: dict, root) -> tuple:
    tbl = Path(root or P.WT) / "scripts/_rl_recipe.sh"
    return (str(tbl.resolve()), tbl.stat().st_mtime_ns, tuple(names), tuple(sorted(env.items())))


def table_values(names: tuple[str, ...], env: dict, root: str | os.PathLike | None = None) -> dict[str, str]:
    """Canonical launch table values (scripts/_rl_recipe.sh) sourced in tree `root` under exactly `env` (its CTE_PATH
    picks the column); unset reads "", a table that does not source dies — the one Python reader of it. Memoized
    (`_TV_MEMO`)."""
    key = _tv_key(names, env, root)
    if key in _TV_MEMO:
        return dict(_TV_MEMO[key])
    _TV_MEMO[key] = out = _table_values(names, env, root)
    return dict(out)


def _table_values(names: tuple[str, ...], env: dict, root: str | os.PathLike | None = None) -> dict[str, str]:
    sh = "source scripts/_rl_recipe.sh >/dev/null && printf '%s\\0' " + " ".join(f'"${{{n}-}}"' for n in names)
    r = subprocess.run(["bash", "-c", sh], cwd=root or P.WT, env=env, capture_output=True, text=True)
    if r.returncode != 0 or len(vals := r.stdout.split("\0")) != len(names) + 1 or vals[-1] != "":
        raise SystemExit(
            f"[run_inputs] cannot read {names} from the canonical table (rc {r.returncode}): {r.stderr.strip()[-300:]}"
        )
    return dict(zip(names, vals[:-1], strict=True))


def canon_env(b: dict, man: dict, names: tuple[str, ...]) -> dict[str, str]:
    """table_values under this bundle's launch env (launch_env over os.environ; side both → the bi column, else uni)."""
    if b["side"] not in SIDES:
        raise ValueError(f"side {b['side']!r} — only {SIDES}")
    env = dict(
        os.environ,
        **dict(kv.replace("$PWD", str(P.WT)).split("=", 1) for kv in launch_env(b, man)),
        CTE_PATH="bi" if b["side"] == "both" else "uni",
    )
    return table_values(names, env)


@contextlib.contextmanager
def _env(**kv):
    """Run an owner that reads the environment (selection.combo_usd_path) under the launch values; the whole
    environment is restored afterwards."""
    saved = os.environ.copy()
    os.environ.update(kv)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def active_objects(lay, side: str, tp) -> list[tuple[str, bool]]:
    """(asset id, articulated) of the main objects a run of `side` spawns for one tree — the entry's rule: one-hand =
    that side's main object; bimanual = each side's main object except an idle side (its slot carries the active
    side's object as a placeholder that is never spawned). asset id = tp.object_asset_id(main_object_id,
    main_articulation) (the path key of usd · meshes). A side that should have a main object and has none dies (the
    entry refuses that clip too)."""
    if side not in ("right", "left") and len(lay.idle_sides()) > 1:
        raise ValueError(f"{lay.tree_dir}: both sides idle — not a training clip")
    out = []
    for s in (side,) if side in ("right", "left") else [s for s in ("right", "left") if s not in lay.idle_sides()]:
        if (oid := lay.main_object_id(s)) is None:
            raise ValueError(f"{lay.tree_dir}: no {s} main object (objects[] role=main)")
        out.append((tp.object_asset_id(oid, art := lay.main_articulation(s)), bool(art)))
    return out


def clip_inputs(clip: str, *, combo: str, hand: str, dataset: str, emb: str, side: str, tp=None) -> list[Item]:
    """The files one clip of a run of `side` reads (module doc). Reads the tree's task_info (qpos_layout) — itself an
    input — and, for the scene build's contact readings, its base tree's task_info (a tree without a base, hand or
    MANO source dies here as the scene build dies there); everything else is a path statement."""
    if side not in SIDES or (side == "both") != (emb == "bimanual") or (side != "both" and emb != side):
        raise ValueError(f"{clip}: side {side!r} and emb {emb!r} do not match (both↔bimanual · right/left↔same name)")
    cp = P.ClipPaths(clip, combo, hand, dataset, emb)
    t0 = cp.tree_dir
    td = t0.parent
    out = [Item(td / "task_info.json", "tree", "entry:layout")]
    lay = P.qpos_layout.read(str(td))
    if lay.source.split("|")[0] in ("legacy_scene_xml", "legacy_default") and (td / "scene.xml").is_file():
        out.append(Item(td / "scene.xml", "tree", "entry:layout-legacy"))  # the reader opens it only if present
    sides = ("right", "left") if side == "both" else (side,)
    out += [
        Item(cp.tree_qk, "trajectory", "entry:discovery"),
        Item(t0 / "trajectory_ikrollout.npz", "trajectory", "kit:loader-discovery"),
        Item(t0 / "body_positions.npz", "trajectory", "kit:body-positions"),
    ]
    out += [Item(cp.contact(s), "trajectory", "kit:contact") for s in sides]
    for s in sides:
        out += [Item(cp.wrist(s), "cache", "entry:preflight"), Item(cp.ik(s), "cache", "entry:preflight")]
    out += [Item(cp.wrist(s), "cache", "kit:wrist-table") for s in ("right", "left") if s not in sides]
    if tp is None:
        from dexengine.core import artifact_paths as tp  # noqa: PLC0415
    tpd = tp.get(dataset)
    for oid, art in active_objects(lay, side, tp):
        out += [
            Item(Path(tpd.usd_file(oid, articulated=art)), "object", "entry:preflight"),
            Item(Path(tpd.surface_points_file(oid, articulated=art)), "object", "entry:preflight"),
            Item(Path(tpd.keypoints_file(oid, articulated=art)), "object", "kit:keypoints"),
        ]
    out += [Item(Path(tpd.usd_file(oid, articulated=False)), "object", "kit:support") for oid in support_objects(td)]
    return out + scene_inputs(td, tpd)


def scene_inputs(td: Path, tpd) -> list[Item]:
    """What the scene build reads for one tree beyond the entry, per rule owner: S-float-settle's contact readings
    (float_settle.inputs, reader kit:float-settle); the extra-slot rule's files (kit:extra · extra_slots.env_inputs;
    none without extras; slot USD = usd_file, rigid; on by default); the cooked hull dump of each rigid object of such
    a tree (kit:scene-rules, where it exists)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.data.objects import float_settle as FS  # noqa: PLC0415

    ti = json.loads((td / "task_info.json").read_text(encoding="utf-8"))
    got = XS.env_inputs(ti, str(td))
    out = [Item(Path(f), "base", "kit:float-settle") for f in FS.inputs(td, ti)]
    out += [Item(Path(tpd.usd_file(oid, articulated=False)), "object", "kit:extra") for oid in got["usd_ids"]]
    out += [Item(Path(f), "object", "kit:extra") for f in got["mesh"]]
    out += [Item(Path(f), "trajectory", "kit:extra") for f in got["tree"]]
    if not got["usd_ids"]:
        return out
    rigid = dict.fromkeys(str(o["obj_id"]) for o in ti.get("objects") or [] if o.get("obj_id") and o.get("mesh_dir"))
    for oid in rigid:
        dump = Path(P.data_layout.cooked_hull_file(str(tpd.usd_file(oid, articulated=False))))
        if dump.is_file():
            out.append(Item(dump, "object", "kit:scene-rules"))
    return out


def support_objects(td: Path) -> list[str]:
    """obj_ids of the support objects the env spawns fixed for one tree — the env's rule (_support_rows ·
    clip_supports, every side): the bake stamp's supports (coords — read_aerial_support on an aerial tree, else
    desk_support_discs; none = 0), one per object name, each an objects[] entry. A stamped support that names no
    object dies here as the env dies there."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    ti = json.loads((td / "task_info.json").read_text(encoding="utf-8"))
    with stdio.capture(sys.stderr):  # the readers' notices ([DESK_SUPPORT] …) — stdout is data here
        stamp = C.read_aerial_support(td) if ti.get("scene") == "aerial" else C.desk_support_discs(td)
    names = list(dict.fromkeys(str(s["obj"]) for s in stamp.get("supports") or []))
    by = {str(o.get("name")): o for o in ti.get("objects") or []}
    bad = [n for n in names if n not in by or by[n].get("articulation")]
    if bad:
        raise ValueError(f"{td}: supports {bad} are not in objects or are articulated — the env dies here too")
    return [str(by[n]["obj_id"]) for n in names]


def robot_inputs(arm: str, hand: str, sw: dict[str, str]) -> list[Item]:
    """The robot a run spawns: the combo USD (selection.combo_usd_path under the table's USD variant) + its bake meta,
    and the split R/L articulations when the table's DEXENGINE_BI_SPLIT_ARTIC is 1 (anything but ""/0/1 dies). The
    files the entry gate requires are selection.required_robot_files: a floating hand has no merged USD — its
    per-side split pair IS the robot (no bake meta). Paths only."""
    split = sw["DEXENGINE_BI_SPLIT_ARTIC"]
    if split not in ("", "0", "1"):
        raise SystemExit(f"[run_inputs] DEXENGINE_BI_SPLIT_ARTIC={split!r} — only 0/1 (or unset)")
    from dexengine.robots import selection as rs  # noqa: PLC0415

    with _env(DEXENGINE_COMBO_USD_VARIANT=sw["DEXENGINE_COMBO_USD_VARIANT"]), stdio.capture(sys.stderr):
        usd = rs.combo_usd_path(arm, hand, rs.USD_CACHE_DIR)  # its notice goes to stderr (stdout is data here)
        variant = rs.resolve_usd_variant(arm, hand, rs.USD_CACHE_DIR)
    need = rs.required_robot_files(arm, "usd", usd)
    if need == (Path(usd),):
        out = [
            Item(Path(usd), "robot", "entry:robot-gate"),
            Item(Path(rs.bake_meta_file(usd)), "robot", "entry:robot-gate"),
        ]
    else:
        out = [Item(p, "robot", "entry:robot-gate") for p in need]
    out += [Item(p, "robot", "entry:robot-identity") for p in identity_inputs(arm, hand, variant, rs)]
    if split == "1":
        from dexengine.rl.tasks.scene import split_artic  # noqa: PLC0415

        out += [Item(p, "robot", "kit:split") for p in split_artic.split_usd_pair(usd) if p not in need]
    return out


def identity_inputs(arm: str, hand: str, variant: str, rs) -> list[Path]:
    """What the entry's robot gate (selection.assert_generation_coherent → derived_usd.verify) reads on the data root
    for a derived body (arms.generation_of(arm, variant).derived_from — none for any other body): the derived record
    (derived_usd.provenance_file), the source USD (md5), a render plate's material source USD (md5), and the files the
    arm collider digest opens there — every <mesh> of the URDFs of the hands registered with the arm
    (sim_collider.arm_hands · combo_urdf) that resolves under the data root (paths.urdf_mesh)."""
    from dexengine.core import arms as AS  # noqa: PLC0415

    gen = AS.generation_of(arm, variant)
    if gen is None or gen.derived_from is None:
        return []
    from dexengine.placement.collision import sim_collider as SC  # noqa: PLC0415 — only a derived body needs it
    from dexengine.robots import derived_usd as du  # noqa: PLC0415

    cache, src = rs.USD_CACHE_DIR, gen.derived_from
    out = [du.provenance_file(rs._usd_literal(arm, hand, cache, variant)), Path(rs._usd_literal(arm, hand, cache, src))]
    mat = AS.generation_of(arm, src).material_of
    if mat is not None:
        out.append(Path(rs._usd_literal(arm, hand, cache, mat)))
    # Only the meshes urdf_mesh places on the data root (first part a data layer head — data_layout.data_tops): in a
    # checkout whose code tree IS the data root, "under the root" alone would also take the tracked meshes, so the
    # statement would depend on where it is derived from. A registered shadow root (data_layout.data_shadow) links
    # its data layers to its base and urdf_mesh answers the resolved file — such a mesh is stated at its place on the
    # root (root / the same relative path).
    root, tops = P.main_root().resolve(), P.data_layout.data_tops()
    sh = P.data_layout.data_shadow(str(P.main_root()))
    bases = [root] + ([Path(sh["base"]).resolve()] if sh else [])
    for u in sorted({SC.combo_urdf(arm, hand, src)} | {SC.combo_urdf(arm, h) for h in SC.arm_hands(arm)}):
        for m in ET.parse(str(u)).getroot().iter("mesh"):
            f = P.urdf_mesh(u, m.get("filename"))
            for b in bases:
                if f.is_relative_to(b) and f.relative_to(b).parts[0] in tops:
                    out.append(root / f.relative_to(b))
                    break
    return list(dict.fromkeys(out))


def axis_ledger_paths(a: dict) -> list[str]:
    """Source file paths of one manifest axis; a round-scoped axis (`round:` files) gives its source ledger."""
    if not any(f["path"].startswith("round:") for f in a["files"]):
        return [f["path"] for f in a["files"]]
    led, sep, _ = a["source"].partition(" ∩ round:")
    if not sep or not led.startswith("ledger:"):
        raise SystemExit(f"[ROUND_SCOPE] {a['axis']}: a round source is not of the form ledger:<ledger> ∩ round:<list>")
    return [led.removeprefix("ledger:")]


def axis_ledger(man: dict, axis: str) -> Path:
    """The folded ledger of one manifest axis (datasets.ledger_file — what the entry resolves); the manifest's source
    for that axis must be exactly that file (compared relative to each machine's data root — the manifest holds the
    builder's absolute paths). Named-roster manifests are not stated."""
    if man["mode"] != "ledger":
        raise SystemExit(f"[run_inputs] roster mode {man['mode']} — only ledger-mode manifests are stated")
    lf = RR.ledger_file(axis, man["hand"], root=P.main_root())
    got = [os.path.relpath(f, man["data_root"]) for a in man["axes"] if a["axis"] == axis for f in axis_ledger_paths(a)]
    if got != [os.path.relpath(lf, P.main_root())]:
        raise SystemExit(f"[run_inputs] axis {axis}: manifest sources {got} ≠ ledger {lf}")
    return lf


def roster_inputs(man: dict, b: dict) -> list[Item]:
    """The bundle's roster ledger (the entry resolves it: ledger_key → ledger_file), the base task_info target_object
    reads for each clip and the surface points the family of each object key reads (verify)."""
    axis = RR.ledger_key(b["dataset"], b["emb"], b["variant"])
    if axis != b["axis"]:
        raise SystemExit(f"[run_inputs] {b['id']}: ledger axis {axis} ≠ manifest {b['axis']}")
    out = [Item(axis_ledger(man, axis), "roster", "entry:roster")]
    hd = RR.hand_dir(man["hand"])
    for c in b["clips"]:
        f = ST.target_object_file(c, b["dataset"], b["emb"], hd)
        if f is not None:
            out.append(Item(f, "base", "verify:base"))
    out += [Item(f, "object", "verify:shape") for k in sorted(b["families"]) for f in ST.shape_files(b["dataset"], k)]
    return out


class Stater:
    """States bundles of one manifest; caches the table values per launch env (a per-dataset or per-hand table value
    never reuses another bundle's) and the robot items. `switches_used` = {"side|dataset|emb": table switches} of what
    was stated."""

    def __init__(self, man: dict):
        from dexengine.core import artifact_paths  # noqa: PLC0415

        self.man = man
        self.root = P.main_root()
        self.tp = artifact_paths
        self._sw: dict[tuple, dict[str, str]] = {}
        self._robot: dict[tuple, list[Item]] = {}
        self.switches_used: dict[str, dict[str, str]] = {}

    def switches(self, b: dict) -> dict[str, str]:
        """The canonical table switches (CANON_SWITCHES) of one bundle's launch env."""
        b = dict(b, hand=self.man["hand"])
        key = tuple(launch_env(b, self.man))
        if key not in self._sw:
            sw = canon_env(b, self.man, CANON_SWITCHES)
            if sw["DEXENGINE_ARM_IK_DIR_SUFFIX"] != P.ik_suffix():
                raise SystemExit(
                    f"[run_inputs] IK suffix — canonical table {sw['DEXENGINE_ARM_IK_DIR_SUFFIX']!r} ≠ this "
                    f"process {P.ik_suffix()!r} (paths.ik_suffix) — it would state another cache than "
                    "the launch reads"
                )
            self._sw[key] = sw
            self.switches_used[f"{b['side']}|{b['dataset']}|{b['emb']}"] = sw
        return self._sw[key]

    def rel(self, p: Path) -> str:
        """Path relative to the data root; an input outside it dies."""
        r = os.path.relpath(p, self.root)
        if r.startswith(".."):
            raise InputMissingError(f"[run_inputs] input {p} outside the data root — the GPU host cannot place it")
        return r

    def bundle(self, b: dict) -> dict[str, tuple[str, str]]:
        """{relative path: (kind, reader)} of one bundle, sorted by path. The first reader wins for a path two
        consumers share (e.g. two clips' one object)."""
        tr = self.rel(P.trees_dir(b["combo"], b["dataset"], b["emb"]))
        if tr != b["tree_root"]:
            raise SystemExit(f"[run_inputs] {b['id']}: tree root {tr} ≠ manifest {b['tree_root']}")
        sw = self.switches(b)
        rk = tuple(sorted(sw.items()))
        # The package resolvers fall back to the working folder without DEXENGINE_MAIN_ROOT and paths.main_root to the
        # code tree; they are made to agree here (the statement must not depend on where it is derived from).
        with _env(DEXENGINE_MAIN_ROOT=str(self.root)):
            if rk not in self._robot:
                self._robot[rk] = robot_inputs(self.man["arm"], self.man["hand"], sw)
            from dexengine.record.looks import look_assets as LA  # noqa: PLC0415

            desk = [Item(p, "look", "kit:desk") for p in LA.willow_files()]
            items = roster_inputs(self.man, b) + self._robot[rk] + desk
            kw = {k: b[k] for k in ("combo", "dataset", "emb", "side")} | {"hand": self.man["hand"], "tp": self.tp}
            for c in b["clips"]:
                items += clip_inputs(c, **kw)
            # kit:extra last (stable): an entry-read object that is also some clip's extra keeps its entry reader.
            items.sort(key=lambda it: it.reader == "kit:extra")
            items += [
                Item(p, "layer", "launch:layer-root")
                for p in P.layer_root_records(b["dataset"], b["combo"], b["emb"], [it.path for it in items])
            ]
        return self._table(items, render=False)

    def render(self, b: dict) -> dict[str, tuple[str, str]]:
        """{relative path: (kind, reader)} the render of one bundle adds (RENDER_READERS): the plate (robot_inputs under
        arms.plate_variant), look D, and per clip F0_REPAIR.json, render-only inputs and object overlays."""
        from dexengine.core import arms as AS  # noqa: PLC0415
        from dexengine.record.looks import look_assets as LA  # noqa: PLC0415
        from dexengine.record.scene import render_only as RO  # noqa: PLC0415

        hand, arm = self.man["hand"], self.man["arm"]
        sw = dict(self.switches(b), DEXENGINE_COMBO_USD_VARIANT=AS.plate_variant(arm, hand))
        with _env(DEXENGINE_MAIN_ROOT=str(self.root)):
            items = [Item(it.path, it.kind, "render:plate") for it in robot_inputs(arm, hand, sw)]
            items += [Item(LA.mirror_root() / f, "look", "render:look") for f in LA.render_files()]
            for c in b["clips"]:
                td = P.ClipPaths(c, b["combo"], hand, b["dataset"], b["emb"]).tree_dir.parent
                items += [Item(td / "F0_REPAIR.json", "tree", "render:f0-meta")] * (td / "F0_REPAIR.json").is_file()
                kw = {k: b[k] for k in ("combo", "dataset", "emb", "side")} | {"hand": hand, "tp": self.tp}
                spawn = [i.path for i in clip_inputs(c, **kw) if i.reader in RENDER_SPAWNED and i.path.suffix == ".usd"]
                ro, ro_usds = RO.render_inputs(json.loads((td / "task_info.json").read_text(encoding="utf-8")))
                items += [Item(f, "object", "render:render-only") for f in ro]
                ov = [self.tp.render_usd_file(u) for u in spawn + ro_usds]  # + what each overlay opens
                items += [Item(Path(f), "object", "render:object") for o in ov for f in (o, *self.tp.overlay_assets(o))]
        return self._table(items, render=True)

    def _table(self, items: list[Item], *, render: bool) -> dict[str, tuple[str, str]]:
        """Items → {relative path: (kind, reader)} sorted by path; the first reader wins for a path two consumers
        share. A render reader in a bundle's set (or the reverse) dies — the two never mix."""
        out: dict[str, tuple[str, str]] = {}
        for it in items:
            if it.kind not in KINDS or it.reader not in READERS or (it.reader in RENDER_READERS) != render:
                raise SystemExit(f"[run_inputs] unknown kind/reader {it}")
            out.setdefault(self.rel(it.path), (it.kind, it.reader))
        return dict(sorted(out.items()))


def manifest_level(man: dict, axes: set[str] | None = None) -> dict[str, tuple[str, str]]:
    """The inputs of the manifest as a whole, not of one bundle: every axis ledger (the manifest check re-hashes each).
    axes = only those axes' ledgers (rl_bundle's test bundle); None = every axis. An axis the manifest does not have
    dies."""
    if axes is not None and (bad := sorted(set(axes) - {a["axis"] for a in man["axes"]})):
        raise SystemExit(f"[run_inputs] axes not in the manifest {bad}")
    root = P.main_root()
    return dict(
        sorted(
            (os.path.relpath(axis_ledger(man, a["axis"]), root), ("roster", "verify:sources"))
            for a in man["axes"]
            if axes is None or a["axis"] in axes
        )
    )


def dataset_level(man: dict, bundles: list[dict] | None = None) -> dict[str, tuple[str, str]]:
    """{relative path: (kind, reader)} of the dataset-level inputs of the bundles' datasets (all the manifest's when
    None) — files a dataset's rules open whole, not per tree (coords.dataset_files — OakInk2's official part tree json,
    opened by the extra-slot rule). Kept out of a bundle's set: they would mark every bundle of the dataset changed
    though no tree changed."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    root = P.main_root()
    dss = sorted({b["dataset"] for b in (man["bundles"] if bundles is None else bundles)})
    return dict(
        sorted((os.path.relpath(str(f), root), ("object", "kit:extra")) for ds in dss for f in C.dataset_files(ds))
    )


def digest(rels) -> str:
    """sha256 over the sorted relative paths (one per line)."""
    return hashlib.sha256("".join(r + "\n" for r in sorted(rels)).encode()).hexdigest()


def expected(man: dict, bundles: list[dict]) -> dict[str, tuple[int, str]]:
    """{bundle id: (n, digest)} re-derived here for the given bundles (the runner's side of the input check)."""
    st = Stater(man)
    return {b["id"]: (len(s), digest(s)) for b in bundles for s in (st.bundle(b),)}
