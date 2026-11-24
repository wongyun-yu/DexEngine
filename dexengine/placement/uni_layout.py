#!/usr/bin/env python3
"""Add an idle hand block to a one-hand emb tree (one hand block) so that it has the bimanual layout.

Why: the stage-2 chain (bake, wrist, IK, cross repair, contact, gate, RL env) knows only one bimanual layout of the
hand blocks, [right 0:H][left H:2H][objects 2H+7k]. One-hand datasets have a single hand block and would die
everywhere. To run the missing hand through the same path as an idle hand (fixed arm, fingers at rest) the tree must
have the bimanual layout first; the added block's virtual wrist is filled by the coarse-path DP (idle_rest) with the
FK wrist pose of the fixed arm.

Pieces:
  · Hand MJCF block (the hand <mesh> lines + the hand root body subtree — the root name is the tree's qpos_layout
    hand block, `hand_root`) = `template_path(<hand>, <side>)`, extracted once from a reference bimanual scene
    (`--extract-for`), with hand_asset_sha256_xml3 in its header (compared with the tree's value at injection). The
    file name `<side>_palm_link.xml` is a fixed name, unrelated to the root body name.
  · Block order = the bimanual convention (right, left, objects). Inserted column = `insert_col(layout, side, H)`.
  · Arrays = `inject_arrays` (a [virtual wrist 0 · finger rest pose hand_surface.rest_finger_q] block into
    qpos/qpos_rollout, a zero block into qvel), statement = `layout_after` / `task_info_after`, oracle =
    `verify_scene` (mujoco compile: nq/nv growth, the added hand's first joint address, every old joint address).
  · The left entries of <contact>/<actuator>/<equality> are not added — FK, render and IK need the bodies only and
    the idle hand never moves.
The bake (bake/combo.py) calls these functions in memory.

CLI:
  --extract-for <hand>   extract both templates from the reference bimanual scene (reference_scene)
  --check <hand>         rc 0 when both templates exist and match the registered rig, else rc 2 and what to run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError, input_missing
from dexengine.core.repo import HAND_ASSETS_DIR
from dexengine.core.repo import REPO_ROOT as WT

SIDES = ("right", "left")
_BODY_OPEN = re.compile(r"<body\b[^>]*(?<!/)>")


def template_dir(hand_dir: str) -> Path:
    """Hand spec directory (e.g. wuji) → template folder `assets/robots/hands/<hand id>/scene_blocks`. The
    hand id comes from the registry (`paths.hand_id`); an unknown spec directory dies there."""
    from dexengine.core import paths as _P  # noqa: PLC0415

    return HAND_ASSETS_DIR / _P.hand_id(hand_dir) / "scene_blocks"


def hand_root(ti: dict, side: str) -> str:
    """The root body name of that side's hand subtree = the hand block name the tree's qpos_layout states (it differs
    per hand: wuji `right_palm_link` · dg5f_s `right_link_mount`). No hand block on that side dies."""
    for e in ti.get("qpos_layout") or []:
        if e.get("kind") == "hand" and e.get("side") == side:
            return str(e["name"])
    raise ValueError(f"qpos_layout has no {side} hand block — the hand root body name is unknown")


def block_root(body: list[str]) -> str:
    """Root body name of a template body block (the first `<body name=…>`)."""
    m = re.search(r'<body name="([^"]+)"', body[0] if body else "")
    if not m:
        raise ValueError("the first line of the template body block is not <body name=…>")
    return m.group(1)


def _body_span(lines: list[str], root: str) -> tuple[int, int]:
    """(first line, last line) of the `<body name="<root>"` subtree — counts nesting to find its closing </body>."""
    start = next((i for i, ln in enumerate(lines) if f'<body name="{root}"' in ln), None)
    if start is None:
        raise ValueError(f"no {root} body")
    depth = 0
    for i in range(start, len(lines)):
        depth += len(_BODY_OPEN.findall(lines[i])) - lines[i].count("</body>")
        if depth == 0:
            return start, i
    raise ValueError(f"the {root} subtree is not closed")


def extract_block(ref_scene: Path, side: str) -> str:
    """The template text of the <side> hand: its <mesh> lines and body subtree from a reference bimanual scene."""
    text = ref_scene.read_text()
    lines = text.split("\n")
    # Hand meshes only — named <side>_… with a file under robots/. Object meshes also use `<side>_visual`/`<side>_0`
    # names, so picking by name alone would carry the reference clip's objects into the template.
    meshes = [ln for ln in lines if re.search(rf'<mesh\s+name="{side}_', ln) and 'file="robots/' in ln]
    if not meshes:
        raise ValueError(f'{ref_scene}: no {side} hand mesh (<mesh name="{side}_… file="robots/…>)')
    ti = json.loads((ref_scene.parent / "task_info.json").read_text())
    root = hand_root(ti, side)
    s, e = _body_span(lines, root)
    sha = ti.get("hand_asset_sha256_xml3")
    if not sha:
        raise ValueError(f"{ref_scene.parent}: task_info.hand_asset_sha256_xml3 missing")
    other = [ln for ln in lines[s : e + 1] if re.search(r'name="(right|left)_', ln) and f'"{side}_' not in ln]
    if other:
        raise ValueError(f"the subtree holds names of the other hand: {other[:2]}")
    head = (
        f"<!-- uni_layout template side={side} root={root} hand_asset_sha256_xml3={sha} "
        f"source={_source_label(ref_scene)} -->\n"
    )
    return (
        head
        + "<uni_block>\n<meshes>\n"
        + "\n".join(meshes)
        + "\n</meshes>\n<body_block>\n"
        + "\n".join(lines[s : e + 1])
        + "\n</body_block>\n</uni_block>\n"
    )


def _source_label(p: Path) -> str:
    """The reference scene for the template header: relative to the code tree or the data root, else its name under
    `<outside>/` (a header records no absolute path)."""
    from dexengine.core import paths as _P  # noqa: PLC0415

    for root in (WT, _P.main_root()):
        if p.is_relative_to(root):
            return str(p.relative_to(root))
    return f"<outside>/{p.name}"


def template_path(hand_dir: str, side: str) -> Path:
    """The template file of one hand side (extraction and loading both read it)."""
    return template_dir(hand_dir) / f"{side}_palm_link.xml"


def templates_missing(hand_dir: str) -> list[Path]:
    """The hand's one-hand scene templates that do not exist (both are needed to bake a one-hand axis)."""
    return [template_path(hand_dir, s) for s in SIDES if not template_path(hand_dir, s).is_file()]


def template_sha(p: Path) -> str:
    """The rig identity a template header states (`hand_asset_sha256_xml3=`) — "" when absent."""
    m = re.search(r"hand_asset_sha256_xml3=(\w+)", p.read_text())
    return m.group(1) if m else ""


def templates_stale(hand_dir: str) -> list[Path]:
    """Existing templates extracted with ANOTHER rig than the registered one (header identity ≠ the hand's registered
    rig `paths.hand_rig_sha`). Templates carry hand sites (fingertips, palm): injecting an old one after a rig change
    would bake one-hand trees with the old sites. Re-extract with `--extract-for <hand>` once the new rig's base layer
    exists. `hand_dir` = hand id or spec directory (normalized by hand_rig_sha); an unknown name is a KeyError."""
    from dexengine.core import paths as _P  # noqa: PLC0415

    reg = _P.hand_rig_sha(hand_dir)
    return [p for p in (template_path(hand_dir, s) for s in SIDES) if p.is_file() and template_sha(p) != reg]


def rig_mismatch_error(p: Path) -> RuntimeError:
    """The error of loading a stale template (header rig identity ≠ registered rig) — raised by load_block."""
    sha = template_sha(p)[:12] or "none"
    return RuntimeError(
        f"[uni_layout] template {p} has rig identity {sha} ≠ the registered rig — the rig changed. "
        "Build the new rig's base layer, then re-extract with `uni_layout.py --extract-for <hand>`"
    )


def reference_scene(hand: str) -> Path:
    """The reference bimanual scene to extract templates from: of the hand's base-layer bimanual scenes, the first by
    dataset name and clip name that states hand blocks for both sides. The hand blocks are the same in every scene of
    the hand (the header's hand_asset_sha256_xml3 checks it). None dies — stage 1 builds the base layer first."""
    from dexengine.core import paths as _P  # noqa: PLC0415

    spec = _P.hand_spec_dir(hand)
    data = _P.main_root() / "data"  # the base layer lives under the data root (not the code tree)
    for ds in sorted(p.name for p in data.iterdir() if (p / "base" / spec / "bimanual").is_dir()):
        for d in sorted((data / ds / "base" / spec / "bimanual").iterdir()):
            ti_p, sc = d / "task_info.json", d / "scene.xml"
            if not (ti_p.is_file() and sc.is_file()):
                continue
            sides = {
                e.get("side") for e in json.loads(ti_p.read_text()).get("qpos_layout") or [] if e.get("kind") == "hand"
            }
            if set(SIDES) <= sides:
                return sc
    raise InputMissingError(
        f"[uni_layout] ✗ hand {hand}: no reference bimanual scene (data/*/base/{spec}/bimanual) — "
        "run stage 1 (base layer) first"
    )


def load_block(hand_dir: str, side: str) -> tuple[str, list[str], list[str]]:
    """(sha, mesh lines, body lines) of one template. The root body name is `block_root(body lines)`. `hand_dir` =
    hand id or spec directory."""
    p = template_path(hand_dir, side)
    if not p.is_file():
        raise FileNotFoundError(
            f"[uni_layout] template missing: {p} — create it with `uni_layout.py --extract-for <hand>`"
        )
    if p in templates_stale(hand_dir):
        raise rig_mismatch_error(p)
    text = p.read_text()
    sha = template_sha(p)
    meshes = re.search(r"<meshes>\n(.*?)\n</meshes>", text, re.S).group(1).split("\n")
    body = re.search(r"<body_block>\n(.*?)\n</body_block>", text, re.S).group(1).split("\n")
    return sha, meshes, body


def insert_col(hands: dict[str, int], side: str, H: int) -> int:
    """First column of the added hand block — keeps the bimanual convention (right, left, objects)."""
    if side in hands:
        raise ValueError(f"the {side} hand exists already: {hands}")
    if side == "left":
        return int(hands["right"]) + H  # right after the right block
    return int(hands["left"])  # right before the left block (= 0)


def inject_scene_text(text: str, side: str, meshes: list[str], body: list[str], other_root: str) -> str:
    """Scene text with the template's meshes and body added. `other_root` = the root body of the other hand already in
    the tree (hand_root(tree task_info, other side)); the new hand's root comes from the template (block_root)."""
    lines = text.split("\n")
    new_root = block_root(body)
    if any(f'name="{new_root}"' in ln for ln in lines):
        raise ValueError(f"the scene has {new_root} already")
    ia = next(i for i, ln in enumerate(lines) if "</asset>" in ln)
    lines[ia:ia] = meshes
    s, e = _body_span(lines, other_root)
    at = e + 1 if side == "left" else s
    lines[at:at] = body
    return "\n".join(lines)


def inject_arrays(z: dict, col: int, H: int, nq_old: int, nv_old: int, hand: str) -> dict:
    """Insert a block of width H = [virtual wrist 0 ×6 · finger rest pose (hand_surface.rest_finger_q)] into
    qpos/qpos_rollout (width nq_old) and a zero block into qvel (width nv_old) at column col. Hand joints are all
    1-dof, so qvel columns = qpos columns (object blocks come after the hands, none before col). The virtual wrist is
    filled by the coarse-path DP (idle_rest) with the FK of the fixed arm."""
    from dexengine.placement.collision import hand_surface as HSF  # noqa: PLC0415

    rest = HSF.rest_finger_q(hand)
    if len(rest) != H - 6:
        raise ValueError(f"[uni_layout] hand width {H} − virtual wrist 6 ≠ fingers {len(rest)} ({hand})")
    out = dict(z)
    for k, v in z.items():
        v = np.asarray(v)
        if v.ndim == 2 and k in ("qpos", "qpos_rollout") and v.shape[1] == nq_old:
            blk = np.zeros((len(v), H), v.dtype)
            blk[:, 6:] = rest
            out[k] = np.concatenate([v[:, :col], blk, v[:, col:]], axis=1)
        elif v.ndim == 2 and k == "qvel" and v.shape[1] == nv_old:
            out[k] = np.concatenate([v[:, :col], np.zeros((len(v), H), v.dtype), v[:, col:]], axis=1)
    return out


def layout_after(entries: list[dict], side: str, col: int, H: int, root: str) -> list[dict]:
    """qpos_layout entries after inserting a hand block of width H at col."""
    new = []
    for e in entries:
        e2 = dict(e)
        if int(e2["start"]) >= col:
            e2["start"] = int(e2["start"]) + H
        new.append(e2)
    new.append({"name": root, "start": col, "width": H, "kind": "hand", "side": side})
    return sorted(new, key=lambda e: int(e["start"]))


def task_info_after(ti: dict, side: str, col: int, H: int, root: str) -> dict:
    """Move EVERY column statement of task_info — qpos_layout (start) · objects[].qpos_start ·
    articulation.qpos_index · nq · hands. objects[] owns the main object column (qpos_layout.read reads main from
    objects[]) — moving only qpos_layout would make the contact derivation read the old column. Slots
    (= (main − hands·H)/7) are unchanged: the hand count and the columns grow together."""
    out = dict(ti)
    out["qpos_layout"] = layout_after(ti["qpos_layout"], side, col, H, root)
    objs = []
    for o in ti.get("objects") or []:
        o2 = dict(o)
        if o2.get("qpos_start") is not None and int(o2["qpos_start"]) >= col:
            o2["qpos_start"] = int(o2["qpos_start"]) + H
        a = o2.get("articulation")
        if a and a.get("qpos_index") is not None and int(a["qpos_index"]) >= col:
            a = dict(a)
            a["qpos_index"] = int(a["qpos_index"]) + H
            o2["articulation"] = a
        objs.append(o2)
    out["objects"] = objs
    out["nq"] = int(ti["nq"]) + H
    out["hands"] = [s for s in SIDES if s in {e.get("side") for e in out["qpos_layout"] if e.get("kind") == "hand"}]
    out["idle_hand_injected"] = [side]
    out["uni_layout_nq_before"] = int(ti["nq"])
    return out


def verify_scene(
    xml_path: Path, side: str, col: int, H: int, nq_old: int, nv_old: int, old_joint_adr: dict[str, int]
) -> None:
    """Mujoco compile oracle — the statement (layout) and the measurement (model joint addresses) must agree."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    if m.nq != nq_old + H or m.nv != nv_old + H:
        raise RuntimeError(f"[uni_layout] compiled nq/nv={m.nq}/{m.nv} ≠ expected {nq_old + H}/{nv_old + H}")
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_pos_x")
    if j < 0 or int(m.jnt_qposadr[j]) != col:
        raise RuntimeError(
            f"[uni_layout] {side}_pos_x joint address {int(m.jnt_qposadr[j]) if j >= 0 else None} ≠ {col}"
        )
    for name, adr in old_joint_adr.items():
        j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        want = adr + (H if adr >= col else 0)
        if j2 < 0 or int(m.jnt_qposadr[j2]) != want:
            raise RuntimeError(
                f"[uni_layout] old joint {name} address {int(m.jnt_qposadr[j2]) if j2 >= 0 else None} ≠ {want}"
            )


def joint_adr(xml_path: Path) -> dict[str, int]:
    """Joint name → qpos address of a compiled scene."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): int(m.jnt_qposadr[j]) for j in range(m.njnt)}


def main() -> int:
    """CLI: --extract-for <hand> · --check <hand> (module doc)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--extract-for",
        default="",
        metavar="HAND",
        help="extract both templates of the hand from the reference bimanual scene (reference_scene)",
    )
    ap.add_argument(
        "--check",
        default="",
        metavar="HAND",
        help="rc 0 when both templates of the hand exist and match its rig, else say what to run and rc 2",
    )
    a = ap.parse_args()
    if not (a.check or a.extract_for):
        ap.error("one of --extract-for · --check is required")
    from dexengine.core import paths as _P  # noqa: PLC0415

    hand = a.check or a.extract_for
    spec = _P.hand_spec_dir(hand)
    if a.extract_for:
        ref = reference_scene(hand)
        for side in SIDES:
            out = template_path(spec, side)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(extract_block(ref, side))
            os.replace(tmp, out)
            print(f"[uni_layout] template → {out} ({out.stat().st_size} B · reference {ref.relative_to(WT)})")
    miss = templates_missing(spec)
    if miss:
        print(
            input_missing(
                f"[UNI_TEMPLATE] ✗ hand {hand}: one-hand scene templates missing "
                f"{[str(m.relative_to(WT)) for m in miss]} — `python3 -m dexengine.placement.uni_layout "
                f"--extract-for {hand}`"
            )
        )
        return 2
    stale = templates_stale(spec)
    if stale:
        print(
            f"[UNI_TEMPLATE] ✗ hand {hand}: one-hand scene templates extracted with an old rig "
            f"{[str(m.relative_to(WT)) for m in stale]} — build the new rig's base layer (stage 1), then "
            f"`python3 -m dexengine.placement.uni_layout --extract-for {hand}`"
        )
        return 2
    print(f"[UNI_TEMPLATE] hand {hand}: {'·'.join(SIDES)} present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
