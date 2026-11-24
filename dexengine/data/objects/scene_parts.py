#!/usr/bin/env python3
"""Derive the convex part lists in scene*.xml (`…/convex/<n>.obj`) from each object's current decomposition.

Why: retargeting (the scene builder, dexengine.retarget.scene) freezes the convex/ file list by number when it builds
the scene, and
base_adapter copies that list into base. When an object is decomposed again and its part count changes (grab
stapler 11 → 12 · hrdexdb black_kitchen_roll_holder 10 → 14 · coffee_tin 15 → 14), the scene
  - with fewer parts points at a missing part (14.obj) and MuJoCo cannot open it; stage 1 step 2 stops with rc 3
  - with more or changed parts loads only a subset of the new parts, so the MuJoCo object shape (2d · 2e · f0) differs
    from the judge USD (silently wrong)

Rules
  - The part set belongs to the decomposition, `decompose.part_files(<convex folder>)`. The scene is a statement, the
    folder the measurement.
  - **One derivation point**: base_layer.clip.adapt_clip (retarget layer → base). The retarget layer (delivery, inputs)
    is not edited. A base scene that differs from the current decomposition is stale (stale_reason, called by
    base_layer.stale.tree_stale_reason), and stage 1 step 2 rebuilds it.
  - Later points that **move** scenes (pieces segment.build · f0 shadow rewrite_scene_dirs · f0 repair copy
    f0.repair_write · joint conversion
    oakink2.artic_convert) call verify_* before moving; a difference is fatal (SceneParts). A difference means the upper
    layer (base) is stale; lower layers do not patch it.
  - Scope = folders the scene **already lists as convex parts** (CONVEX_PART_RE). Scenes that collide on visual meshes
    (most grab · oakink2 · h2o) are left alone even when a convex folder exists; that choice belongs to retargeting.
  - Part order = the scene builder's (scene_order: file-name string order 0,1,10,11,…,2). The oracle
    `--oracle` measures
    're-stamping with the same parts = the original' on every real scene.
  - The parts' `_decomp.json` fingerprints are not checked here; whether a decomposition is canonical is filtered by the
    bake (objects.usd_bake `[BAKE_CONVEX]`). base loads the part files **present now** there (the files the bake
        takes).

Reconstruction rule (a rule, not a table; shapes that do not fit fail)
  Elements of one part = `<mesh name=P+n file=D/n.obj>` · `<geom name=G+n mesh=P+n>` · the `<pair geom1|geom2>` that
  name that geom (with `name`: `<geom1>_<geom2>`). Elements in one parent that differ only in the part number form a
  family; the family is re-stamped with the new numbers at the position of its first element. A pair between the parts
  of two objects (hot3d extra × extra · right × left) is the product of the two numbers. Failing shapes: a name not
  ending in a number · not exactly one mesh/geom per part · a family not covering every old number exactly once · a part
  name referenced outside mesh/geom/pair · a part element with children · a missing or empty folder.

Marker: `[SCENE_PARTS] <what> <folder> scene=<old count> now=<current count>` (per derived folder, printed by
base_adapter).
CLI (python3; no retarget venv needed):
  python3 dexengine/data/objects/scene_parts.py --scan <folders…>    check every scene*.xml below · rc 1 = any mismatch
  python3 dexengine/data/objects/scene_parts.py --oracle <folders…>  + re-stamping with the same parts = original
      (checks the rule)
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

#: Scene files of a tree folder; points that move scenes read this name instead of restating the tuple.
SCENE_XMLS = ("scene.xml", "scene_eq.xml")
#: Convex part mesh file: the last folder is `convex` or `convex_<part>` and the file name is a number.
CONVEX_PART_RE = re.compile(r"^(?P<dir>(?:.*/)?convex(?:_[^/]+)?)/(?P<stem>\d+)\.obj$")
_MESH_TAG_RE = re.compile(r"<mesh\b[^>]*>")
_FILE_ATTR_RE = re.compile(r'\bfile="([^"]*)"')
_NAME_ATTR_RE = re.compile(r'\bname="([^"]*)"')
_MESHDIR_RE = re.compile(r'<compiler\b[^>]*\bmeshdir="([^"]*)"')


class SceneParts(RuntimeError):
    """The scene's part list differs from the current decomposition, or the reconstruction rule does not fit that
    scene."""


def _decompose():
    """Decomposition owner module (dexengine/retarget/decompose/), loaded through decompose_objects only."""
    from dexengine.data.objects import (
        decompose as decompose_objects,  # noqa: PLC0415 — same folder, stdlib only (loaded late; the CLI path does not use it)
    )

    return decompose_objects.decompose_module()


def scene_order(files) -> list[str]:
    """Part order inside a scene: the scene builder's lexicographic order of the `*.obj` file names
    (0, 1, 10, 11, …, 2)."""
    return sorted(files)


def current_parts(convex_dir: Path) -> list[str]:
    """Current decomposition of the folder (part file names, scene order). A missing folder or no parts is an error."""
    if not Path(convex_dir).is_dir():
        raise SceneParts(f"no convex folder: {convex_dir}")
    try:
        files = _decompose().part_files(str(convex_dir))
    except ValueError as e:  # a .obj that is not a number does not have the decomposition's shape
        raise SceneParts(f"{convex_dir}: part names are not numbers ({e})") from e
    if not files:
        raise SceneParts(f"convex folder has no parts: {convex_dir}")
    return scene_order(files)


def assets_root_of(xml_text: str, scene_dir: Path) -> Path:
    """Location the scene's `<compiler meshdir>` points at (relative paths resolve against the scene folder)."""
    m = _MESHDIR_RE.search(xml_text)
    if m is None:
        raise SceneParts(f"{scene_dir}: no <compiler meshdir> — cannot resolve the part folder")
    md = m.group(
        1
    )  # resolved to a real path as MuJoCo does ('..' in a linked folder is relative to the link target, not textual)
    return Path(os.path.realpath(md if os.path.isabs(md) else os.path.join(scene_dir, md)))


def _prefix(name: str, stem: str, what: str) -> str:
    if not name or not name.endswith(stem) or len(name) == len(stem):
        raise SceneParts(f"{what} name {name!r} does not end in part number {stem} — outside the reconstruction rule")
    return name[: -len(stem)]


def part_refs(xml_text: str) -> dict[tuple[str, str], list[str]]:
    """Convex parts the scene lists: {(folder (relative to meshdir), mesh name prefix): [part file names, scene order]}.

    A group is folder × prefix, since two bodies can list one object folder (a hocap main object also under extra ·
    duplicates before merging)."""
    out: dict[tuple[str, str], list[str]] = {}
    if "convex" not in xml_text:
        return out
    for tag in _MESH_TAG_RE.findall(xml_text):
        f = _FILE_ATTR_RE.search(tag)
        m = CONVEX_PART_RE.match(f.group(1)) if f else None
        if m:
            n = _NAME_ATTR_RE.search(tag)
            key = (m["dir"], _prefix(n.group(1) if n else "", m["stem"], "mesh"))
            out.setdefault(key, []).append(f"{m['stem']}.obj")
    return out


def _compare(xml_text: str, assets_root: Path):
    """(mismatches [(group, scene parts, current parts)], groups without a folder [(group, folder)]). A group is a
    mismatch
    when the sets differ or a part is listed twice. A folder that **exists** but is empty or has non-number parts
    fails."""
    bad, missing = [], []
    for (d, mp), have in part_refs(xml_text).items():
        key, folder = f"{d} · {mp}*", Path(assets_root) / d
        if not folder.is_dir():
            missing.append((key, folder))
            continue
        now = current_parts(folder)
        if sorted(have) != sorted(now):
            bad.append((key, have, now))
    return bad, missing


def mismatches(xml_text: str, assets_root: Path) -> list[tuple[str, list[str], list[str]]]:
    """[(group 'folder · prefix', scene parts, current parts)]; strict: a missing folder also fails (derivation · base
    staleness · --scan)."""
    bad, missing = _compare(xml_text, assets_root)
    if missing:
        more = f" and {len(missing) - 1} more" if len(missing) > 1 else ""
        raise SceneParts(f"no convex folder: {missing[0][1]}{more}")
    return bad


#: Folders for which a point that moves scenes has printed 'cannot check' (once per process).
_UNMEASURED: set[str] = set()


def _fmt(what: str, bad) -> str:
    return "; ".join(
        f"{what} {d}: scene {len(h)} parts ≠ current decomposition {len(n)} parts"
        f" (scene only {sorted(set(h) - set(n))[:4]} · decomposition only {sorted(set(n) - set(h))[:4]})"
        for d, h, n in bad
    )


def verify_text(xml_text: str, scene_dir: Path, what: str = "") -> int:
    """Check scene text before moving it (points that move scenes); a part list that **differs** from the current
    decomposition fails (SceneParts). Returns the number of groups checked.

    When this machine has **no** decomposition folder the check is impossible (not a difference): the marker
    `[SCENE_PARTS] cannot check` is printed once per folder and the scene moves as is. Points that move scenes never
    open
    it in MuJoCo (f0 does not read the scene; judge and training shapes come from the USD). Example: TACO base scenes
    list `taco_retarget/processed/taco/assets/objects/*/convex`, which a judging host lacks; derivation and staleness
    are checked strictly on the machine that builds base (mismatches · stale_reason)."""
    if not part_refs(xml_text):
        return 0
    bad, missing = _compare(xml_text, assets_root_of(xml_text, Path(scene_dir)))
    if bad:
        raise SceneParts(
            f"[SCENE_PARTS] ✗ {_fmt(what or str(scene_dir), bad)} — the upper-layer scene (base) is stale: "
            "stage 1 step 2 (base_adapter) rebuilds it from the current decomposition before moving"
        )
    for _key, folder in missing:
        if str(folder) not in _UNMEASURED:
            _UNMEASURED.add(str(folder))
            print(
                f"[SCENE_PARTS] cannot check {folder}: no decomposition folder on this machine — the scene moves as is"
                " (this machine's MuJoCo cannot open it)",
                flush=True,
            )
    return len(part_refs(xml_text)) - len(missing)


def verify_dir(tree: Path) -> int:
    """verify_text on every scene*.xml of a tree folder. Returns the total number of folders checked."""
    n = 0
    for name in SCENE_XMLS:
        p = Path(tree) / name
        if p.is_file():
            n += verify_text(p.read_text(), p.parent, str(p))
    return n


def stale_reason(tree: Path) -> str:
    """Whether a base tree's scene matches the current decomposition: the reason when it differs, "" when it matches.
    Strict: a missing scene.xml or decomposition folder is also stale (measured on the machine that builds base)."""
    if not (Path(tree) / "scene.xml").is_file():
        return "no scene.xml"
    for name in SCENE_XMLS:
        p = Path(tree) / name
        if not p.is_file():
            continue
        text = p.read_text()
        try:
            bad = mismatches(text, assets_root_of(text, p.parent)) if part_refs(text) else []
        except SceneParts as e:
            return f"[SCENE_PARTS] ✗ {p}: {e}"
        if bad:
            return f"[SCENE_PARTS] ✗ {_fmt(str(p), bad)}"
    return ""


# Reconstruction (called only by base_adapter).
def _groups(root: ET.Element):
    """Convex part groups of the scene: {(folder, mesh prefix): {"dir", "mp", "gp", "mesh": {number: element}, "geom":
    {number: element}}}."""
    groups: dict[tuple[str, str], dict] = {}
    for el in root.iter("mesh"):
        m = CONVEX_PART_RE.match(el.get("file") or "")
        if not m:
            continue
        mp = _prefix(el.get("name"), m["stem"], "mesh")
        g = groups.setdefault((m["dir"], mp), {"dir": m["dir"], "mp": mp, "mesh": {}, "geom": {}})
        if m["stem"] in g["mesh"]:
            raise SceneParts(f"{m['dir']} · {mp}*: part {m['stem']} has more than one mesh")
        g["mesh"][m["stem"]] = el
    by_mesh = {el.get("name"): (k, s) for k, g in groups.items() for s, el in g["mesh"].items()}
    for el in root.iter("geom"):
        hit = by_mesh.get(el.get("mesh"))
        if hit is None:
            continue
        k, s = hit
        if s in groups[k]["geom"]:
            raise SceneParts(f"{k[0]} · {k[1]}*: more than one geom references part {s}")
        groups[k]["geom"][s] = el
    for k, g in groups.items():
        if set(g["geom"]) != set(g["mesh"]):
            raise SceneParts(
                f"{k[0]} · {k[1]}*: of {len(g['mesh'])} part meshes, those with a geom: "
                f"{len(g['geom'])} — not one geom per part"
            )
        gps = {_prefix(el.get("name"), s, "geom") for s, el in g["geom"].items()}
        if len(gps) != 1:
            raise SceneParts(f"{k[0]} · {k[1]}*: geom name prefix is not unique {sorted(gps)}")
        g["gp"] = gps.pop()
    return groups


def _members(root: ET.Element, groups: dict):
    """Per element that references parts: (parent, element, refs [(attribute, folder, number)]); references outside the
    rule fail."""
    mesh_tok = {el.get("name"): (d, s) for d, g in groups.items() for s, el in g["mesh"].items()}
    geom_tok = {el.get("name"): (d, s) for d, g in groups.items() for s, el in g["geom"].items()}
    own = {id(el): ("mesh", d, s) for d, g in groups.items() for s, el in g["mesh"].items()}
    own.update({id(el): ("geom", d, s) for d, g in groups.items() for s, el in g["geom"].items()})
    out = []
    for parent in root.iter():
        for el in list(parent):
            if id(el) in own:
                refs = [own[id(el)]]
            else:
                refs = []
                for k, v in el.attrib.items():
                    if v in mesh_tok:
                        raise SceneParts(
                            f"<{el.tag} {k}={v!r}> references a part mesh — outside the reconstruction rule"
                        )
                    if v in geom_tok:
                        if not (el.tag == "pair" and k in ("geom1", "geom2")):
                            raise SceneParts(
                                f"<{el.tag} {k}={v!r}> references a part geom — outside the reconstruction rule"
                            )
                        refs.append((k,) + geom_tok[v])
            if not refs:
                continue
            if len(el):
                raise SceneParts(
                    f"part element <{el.tag} name={el.get('name')!r}> has children — outside the reconstruction rule"
                )
            if len({d for _k, d, _s in refs}) != len(refs):
                raise SceneParts(
                    f"<{el.tag} name={el.get('name')!r}> references one object's part twice — "
                    "outside the reconstruction rule"
                )
            pair_name = f"{el.get('geom1')}_{el.get('geom2')}"
            if el.tag == "pair" and el.get("name") is not None and el.get("name") != pair_name:
                raise SceneParts(
                    f"<pair name={el.get('name')!r}> is not of the form '<geom1>_<geom2>' — outside the "
                    "reconstruction rule"
                )
            out.append((parent, el, refs))
    return out


def _signature(parent, el, refs) -> tuple:
    ref_attr = {"mesh": ("name", "file"), "geom": ("name", "mesh")}
    skip = set(ref_attr.get(el.tag, ())) | {k for k, _d, _s in refs} | ({"name"} if el.tag == "pair" else set())
    keyed = tuple(sorted((k, v) for k, v in el.attrib.items() if k not in skip))
    return (id(parent), el.tag, tuple((k, d) for k, d, _s in refs), keyed)


def _set_refs(el, refs_new, groups) -> None:
    """Write the new numbers into a cloned element (refs_new = [(attribute, folder, new number)])."""
    for k, d, s in refs_new:
        g = groups[d]
        if k == "mesh":
            el.set("name", g["mp"] + s)
            el.set("file", f"{g['dir']}/{s}.obj")
        elif k == "geom":
            el.set("name", g["gp"] + s)
            el.set("mesh", g["mp"] + s)
        else:
            el.set(k, g["gp"] + s)
    if el.tag == "pair" and el.get("name") is not None:
        el.set("name", f"{el.get('geom1')}_{el.get('geom2')}")


def rederive(root: ET.Element, assets_root: Path, force: bool = False) -> list[dict]:
    """Re-stamp the scene's (ET root) convex part groups with the current decomposition, in place. Returns {dir, scene,
    now}
    per changed folder.

    force=True re-stamps groups with equal sets too (oracle: does the reconstruction rule reproduce the original)."""
    groups = _groups(root)
    if not groups:
        return []
    by_dir = {
        g["dir"]: [os.path.splitext(f)[0] for f in current_parts(Path(assets_root) / g["dir"])] for g in groups.values()
    }
    now = {k: by_dir[g["dir"]] for k, g in groups.items()}
    changed = {d for d, g in groups.items() if force or set(g["mesh"]) != set(now[d])}
    if not changed:
        return []
    families: dict[tuple, list] = {}
    for parent, el, refs in _members(root, groups):
        families.setdefault(_signature(parent, el, refs), []).append((parent, el, refs))
    plan: dict[int, tuple] = {}  # id(first element) → list of new elements · other element ids → None
    for sig, mem in families.items():
        dirs = [d for _k, d in sig[2]]
        if not changed & set(dirs):
            continue
        old = [tuple(s for _k, _d, s in refs) for _p, _e, refs in mem]
        want = list(itertools.product(*(sorted(groups[d]["mesh"]) for d in dirs)))
        if sorted(old) != sorted(want):
            raise SceneParts(
                f"<{sig[1]}> family (folder {dirs}) does not cover every old number exactly once"
                f" ({len(old)} elements · expected {len(want)}) — outside the reconstruction rule"
            )
        tmpl = mem[0][1]
        keys = [k for k, _d in sig[2]]
        new_els = []
        for combo in itertools.product(*(now[d] for d in dirs)):
            c = copy.deepcopy(tmpl)
            _set_refs(c, list(zip(keys, dirs, combo, strict=True)), groups)
            new_els.append(c)
        plan[id(mem[0][1])] = (mem[0][0], new_els)
        for _p, e, _r in mem[1:]:
            plan[id(e)] = None
    for parent in {id(p): p for p, _ in (v for v in plan.values() if v)}.values():
        # family elements share one parent (the signature includes it); new elements go where the first one was
        kids = []
        for ch in list(parent):
            if id(ch) not in plan:
                kids.append(ch)
            elif plan[id(ch)] is not None:
                kids.extend(plan[id(ch)][1])
        parent[:] = kids
    return [
        {
            "dir": groups[k]["dir"],
            "mesh": groups[k]["mp"],
            "scene": scene_order(f"{s}.obj" for s in groups[k]["mesh"]),
            "now": scene_order(f"{s}.obj" for s in now[k]),
        }
        for k in sorted(changed)
    ]


def _canon(root: ET.Element) -> bytes:
    r = copy.deepcopy(root)
    ET.indent(r, space="  ")
    return ET.tostring(r)


def oracle_file(xml_path: Path) -> str:
    """Whether the scene re-stamped with the same parts equals the original (checks the rule): the reason if not, "" if
    so."""
    text = xml_path.read_text()
    if not part_refs(text):
        return ""
    root = ET.fromstring(text)
    before = _canon(root)
    ar = assets_root_of(text, xml_path.parent)
    try:
        if mismatches(text, ar):
            return "mismatch (not an oracle target)"
        rederive(root, ar, force=True)
    except SceneParts as e:
        return f"outside the rule: {e}"
    return "" if _canon(root) == before else "re-stamped scene ≠ original"


def _scenes(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_file():
            out.append(p)
        else:
            out += sorted(q for n in SCENE_XMLS for q in p.rglob(n))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="scene*.xml or folders containing them")
    ap.add_argument(
        "--scan", action="store_true", help="check part lists ↔ current decomposition (rc 1 = any mismatch)"
    )
    ap.add_argument(
        "--oracle", action="store_true", help="+ re-stamping with the same parts = original (checks the rule)"
    )
    a = ap.parse_args()
    if not (a.scan or a.oracle):
        ap.error("--scan or --oracle")
    files = _scenes(a.paths)
    n_ref = n_bad = n_orc = 0
    for f in files:
        text = f.read_text()
        if not part_refs(text):
            continue
        n_ref += 1
        try:
            bad = mismatches(text, assets_root_of(text, f.parent))
        except SceneParts as e:
            bad = [("?", [], [])]
            print(f"[SCENE_PARTS] ✗ {f}: {e}", flush=True)
        if bad and bad[0][0] != "?":
            print(f"[SCENE_PARTS] ✗ {_fmt(str(f), bad)}", flush=True)
        n_bad += bool(bad)
        if a.oracle and not bad:
            why = oracle_file(f)
            if why:
                n_orc += 1
                print(f"[SCENE_PARTS] ✗ oracle {f}: {why}", flush=True)
    print(
        f"[SCENE_PARTS] scenes={len(files)} convex_scenes={n_ref} mismatch={n_bad}"
        + (f" oracle_fail={n_orc}" if a.oracle else ""),
        flush=True,
    )
    return 1 if n_bad or n_orc else 0


if __name__ == "__main__":
    sys.exit(main())
