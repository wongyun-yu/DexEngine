#!/usr/bin/env python3
"""The ``_truncated`` asset variant: hand joint limits clipped to the retargeting (IK) hyperextension cut — the hand
URDFs and the robot+hand merged URDF.

What it makes. Copies of the existing hand URDFs (both hands) and of the existing merged URDF
(`full_robot_gen/<combo dir>/<combo dir>.urdf`) in which only the hand joints' `<limit lower/upper>` values are clipped
by the same rule as the IK model. Every other byte is the original (text substitution plus one marker comment line —
the checks below prove it).
  · hand URDF   → `<original name>_truncated.urdf` (same folder — relative mesh paths still resolve)
  · merged URDF → `full_robot_gen/<combo dir>_truncated/<combo dir>_truncated.urdf` (sibling folder — mesh path text
                  unchanged)
  · ledger      → `truncation_provenance.json` in the merged folder
                  (changed joints old→new · input/output sha256 · rule file · prior · git)

The rule is not here. Which joints are clipped, in which order and how is owned by
`dexengine/retarget/ik/prior.py` (`prior.hyperext_clip_targets` · `prior.hyperext_clip_range` ·
`prior.build_prior_conf`, imported by the hand tracker); this generator calls its window module
`dexengine.retarget.ik.hyperext_limits` with the retargeting interpreter (`paths.require_retarget_py`)
as `-m` from the repository root — the same resolution the retargeting uses for the hand tracker. If the window's
`rule_file` is not under this tree's `dexengine/retarget/ik/`, the run stops.

Checks it runs itself (a mismatch stops the run):
  1. The hand URDF's clipped limits == the same joints' limits in the canonical merged URDF — evidence that the file is
     the merge source. Otherwise name the source with `--hand-urdf` (default HandSpec.vendor_urdf — for every
     registered hand the same document as the merge source).
  2. Output vs original (hand and merged each): with the limit value text blanked and the marker line removed the bytes
     are identical · in the element tree only the changed joints' `limit@lower|upper` differ, with the rule's values.
  3. Clipped asset limits vs the same rule applied to the IK rig xml ranges: difference ≤ RIG_TOL (only the rig xml's
     rounding is allowed).
  4. The output URDF's meshes resolve to the same files as the original (the merged copy from its new place —
     resolution rule = paths.urdf_mesh).

Usage (in a worktree pass the data root as DEXENGINE_MAIN_ROOT):
    python3 dexengine/robots/urdf/truncate.py --hand dg5f_s
    python3 dexengine/robots/urdf/truncate.py --hand wuji2
Then the USD follows the canonical bake path: python scripts/build_robot_usd.py --combos <combo dir>_truncated
→ render body render_body.py → collider revision collider_rev.py --arm <arm> --hand <hand> (that hand's bodies).
Runtime selection: with the hand's HandSpec.sim_finger_limits set to "truncated" (kind → suffix arms.HAND_LIMIT_SUFFIX)
the simulation (training · RL eval · render) spawns the registered generation `<canon><suffix>(_mi)_armhull` (generation
check · derived check · spawn marker [USD_VARIANT]). The judge, IK and bake do not read it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import xml.etree.ElementTree as ET

# isort: split
from dexengine.core import arms
from dexengine.core import hands as hand_specs
from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.core.repo import FULL_ROBOT_URDF_DIR
from dexengine.core.repo import REPO_ROOT as REPO

#: Name of this variant (output folders, files, runtime variant suffix) — owner arms.TRUNCATED_HAND_LIMITS (which hand
#: spawns this body is decided by HandSpec.sim_finger_limits).
SUFFIX = arms.TRUNCATED_HAND_LIMITS
ROBOT_EMB = "bimanual"  # the merged robot carries both hands — the window resolves the side set
#: The rule's window module (prior.py through the IK rig) and the folder its rule file must lie in.
WINDOW = "dexengine.retarget.ik.hyperext_limits"
RULE_DIR = REPO / "dexengine" / "retarget" / "ik"
RIG_TOL = 1e-5  # rad — the IK rig xml writes limits rounded to 6 digits
PROV_NAME = "truncation_provenance.json"
#: Marker put on the output URDF's second line (ASCII — vendor hand URDFs are ASCII, so other tools see no encoding
#: change).
MARK = (
    "<!-- DERIVED by dexengine/robots/urdf/truncate.py: only the hand joint limit lower/upper "
    "values differ, clipped by the IK hyperextension rule (prior.hyperext_clip_range). Ledger: {prov} -->"
)
_VAL = re.compile(r"""(\b(?:lower|upper)=)(["'])[^"']*\2""")


def die(msg: str) -> None:
    raise SystemExit(f"[TRUNC] ✗ {msg}")


def _read(p: pathlib.Path) -> str:
    """The text byte for byte — line endings are kept (some vendor files are CRLF). Write only with `_write`."""
    return pathlib.Path(p).read_bytes().decode("utf-8")


def _write(p: pathlib.Path, text: str) -> None:
    pathlib.Path(p).write_bytes(text.encode("utf-8"))


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _rel(p) -> str:
    """Path for the record — relative to the code tree (REPO) when under it, else relative to the data root when under
    it, else its name under `<outside>/`. The code tree goes first: a worktree lies inside the data root, so the other
    order would record `<scratch>/worktrees/…`."""
    p = pathlib.Path(p).resolve()
    for root in (REPO.resolve(), P.main_root()):
        if root == p or root in p.parents:
            return str(p.relative_to(root))
    return f"<outside>/{p.name}"


def _limits(root: ET.Element) -> dict:
    """Joint name → (lower, upper) — only joints whose `<limit>` states both lower and upper."""
    out = {}
    for j in root.iter("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def _window(spec_dir: str, ranges: dict) -> dict:
    """Run the IK clipping rule (prior.py) in its interpreter and from this tree (the window module); returns the
    rule's JSON output."""
    env = dict(os.environ, MUJOCO_GL="disable")  # the window only compiles the rig — no rendering
    r = subprocess.run(
        [P.require_retarget_py(), "-m", WINDOW, spec_dir, ROBOT_EMB],
        cwd=REPO,
        input=json.dumps(ranges),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if r.returncode != 0:
        die(f"window {WINDOW} failed rc={r.returncode}:\n{r.stderr[-2000:]}")
    out = json.loads(r.stdout)
    rule_file = pathlib.Path(out["rule_file"]).resolve()
    if RULE_DIR not in rule_file.parents:
        die(f"the rule was read from another tree: {rule_file} (expected under {RULE_DIR})")
    return out


def _fmt(v: float) -> str:
    return repr(float(v))


def _rewrite_limits(text: str, new: dict) -> str:
    """Joint name → (lo, hi). Only the lower/upper value text that changes is rewritten, in the one `<limit …>` inside
    that joint element."""
    for name, (lo, hi) in new.items():
        hits = list(re.finditer(rf"""<joint\b[^>]*\bname=(["']){re.escape(name)}\1""", text))
        if len(hits) != 1:
            die(f"joint {name!r}: {len(hits)} elements (expected 1)")
        start = hits[0].start()
        end = text.find("</joint>", start)
        seg = text[start:end]
        lims = list(re.finditer(r"<limit\b[^>]*>", seg))
        if end < 0 or len(lims) != 1:
            die(f"joint {name!r}: {len(lims)} <limit> elements (expected 1)")
        tag = lims[0].group(0)
        for attr, v in (("lower", lo), ("upper", hi)):
            m = re.search(rf"""(\b{attr}=)(["'])([^"']*)\2""", tag)
            if m is None:
                die(input_missing(f"joint {name!r}: <limit> has no {attr}"))
            if float(m.group(3)) != v:
                tag = tag[: m.start(3)] + _fmt(v) + tag[m.end(3) :]
        seg = seg[: lims[0].start()] + tag + seg[lims[0].end() :]
        text = text[:start] + seg + text[end:]
    return text


def _mark_line(text: str, prov: str) -> str:
    """The marker comment line — with the file's own line ending."""
    return MARK.format(prov=prov) + ("\r\n" if "\r\n" in text else "\n")


def _with_mark(text: str, prov: str) -> str:
    mark = _mark_line(text, prov)
    if text.startswith("<?xml"):
        nl = text.index("\n") + 1
        return text[:nl] + mark + text[nl:]
    return mark + text


def _struct_diff(a: ET.Element, b: ET.Element, path: str) -> list:
    """(place, attribute, a, b) where tag, attribute, text or child count differ. Joints and links are placed by
    name."""
    if a.tag != b.tag:
        return [(path, "<tag>", a.tag, b.tag)]
    out = [(path, k, a.get(k), b.get(k)) for k in sorted(set(a.attrib) | set(b.attrib)) if a.get(k) != b.get(k)]
    if (a.text or "").strip() != (b.text or "").strip():
        out.append((path, "<text>", a.text, b.text))
    ca, cb = list(a), list(b)
    if len(ca) != len(cb):
        return out + [(path, "<children>", len(ca), len(cb))]
    for i, (x, y) in enumerate(zip(ca, cb, strict=True)):
        out += _struct_diff(x, y, f"{path}/{x.tag}[{x.get('name', i)}]")
    return out


def _check_only_limits(orig: str, new: str, changed: dict, prov: str, what: str) -> int:
    """Check that the output differs from the original only in limit values. changed = joint → (old (lo,hi), new
    (lo,hi)).
    Returns the number of changed value attributes."""
    mark = _mark_line(orig, prov)
    if new.count(mark) != 1 or _VAL.sub(r"\1\2\2", new.replace(mark, "")) != _VAL.sub(r"\1\2\2", orig):
        die(f"{what}: bytes differ outside the limit value text and the marker line")
    diffs = _struct_diff(ET.fromstring(orig.encode()), ET.fromstring(new.encode()), "robot")
    want = {}
    for j, (old, nw) in changed.items():
        for attr, o, n in (("lower", old[0], nw[0]), ("upper", old[1], nw[1])):
            if o != n:
                want[(j, attr)] = n
    got = {}
    for where, attr, _a, b in diffs:
        m = re.fullmatch(r"robot/joint\[(.+)\]/limit\[\d+\]", where)
        if m is None or attr not in ("lower", "upper"):
            die(f"{what}: a non-limit place differs — {where} @{attr}")
        got[(m.group(1), attr)] = float(b)
    if got != want:
        die(f"{what}: changed limits {sorted(got)} ≠ rule output {sorted(want)}")
    return len(got)


def _meshes(text: str, urdf: pathlib.Path) -> list:
    """(mesh name, actual file or None) by the one URDF mesh rule (P.urdf_mesh), `urdf` being the text's place (for an
    output not written yet, the place it will be written to — `..` is folded textually)."""
    return [
        (fn, f if (f := P.urdf_mesh(urdf, fn)).is_file() else None)
        for fn in re.findall(r"""<mesh\b[^>]*\bfilename=["']([^"']+)""", text)
    ]


def _joint_rows(rows: list) -> dict:
    return {r["joint"]: ((r["lo"], r["hi"]), (r["new_lo"], r["new_hi"])) for r in rows}


def _deg(pair) -> list:
    return [round(math.degrees(v), 6) for v in pair]


def _do_merged(canon: pathlib.Path, out: pathlib.Path, rows: list, prov: str) -> tuple:
    """Canonical merged URDF → clipped copy (output path, text, ledger entry). Nothing is written here — main writes
    after every check. The meshes must resolve to the same files as the original from the new place (sibling folder —
    mesh path text unchanged)."""
    orig = _read(canon)
    changed = {j: v for j, v in _joint_rows(rows).items() if v[0] != v[1]}
    new = _with_mark(_rewrite_limits(orig, {j: v[1] for j, v in changed.items()}), prov)
    n_attr = _check_only_limits(orig, new, changed, prov, "merged URDF")
    got, ref = _meshes(new, out), _meshes(orig, canon)
    bad = [fn for fn, f in got if f is None]
    if bad or [f for _, f in got] != [f for _, f in ref]:
        die(
            f"merged URDF meshes do not resolve like the original from the new place — unresolved {len(bad)} "
            f"e.g. {bad[:3]}"
        )
    print(
        f"[TRUNC] merged URDF: {len(changed)} joints · {n_attr} value attributes changed · meshes "
        f"{len(got)}/{len(got)} resolved "
        f"-> {_rel(out)}"
    )
    return (
        out,
        new,
        {
            "path": _rel(out),
            "source": _rel(canon),
            "source_sha256": _sha(canon),
            "changed_joints": sorted(changed),
            "changed_attrs": n_attr,
            "meshes_resolved": len(got),
        },
    )


def _hand_files(hs, arg: str | None, sides: list) -> list:
    """[(hand URDF, the sides that file holds)]. With `{side}` one file per side, otherwise both hands in one file."""
    # The place of vendor_urdf (repo-relative) follows the same rule as URDF meshes (paths.urdf_mesh): a data-layer head
    # (data/_stage/<hand>) means the data root, anything else this code tree (REPO). The copy is written next to its
    # source, so pinning the data root would overwrite another checkout's tracked files.
    vu = hs.vendor_urdf
    tmpl = arg if arg else (str(P.urdf_mesh(REPO / "vendor_urdf", vu)) if vu else "")
    if not tmpl:
        die(f"hand {hs.name!r}: HandSpec.vendor_urdf is empty — name the source with --hand-urdf")
    if "{side}" in tmpl:
        return [(pathlib.Path(tmpl.format(side=s)).resolve(), [s]) for s in sides]
    return [(pathlib.Path(tmpl).resolve(), list(sides))]


def _do_hand(path: pathlib.Path, file_sides: list, rows: list, ctx: dict) -> tuple:
    """One hand URDF → clipped copy in the same folder (output path, text, ledger entry). Nothing is written here.
    Joint names map through HandSpec.vendor_link_prefix + strip_side (HandSpec owns the vendor spelling)."""
    if not path.is_file():
        die(input_missing(f"hand URDF missing: {path}"))
    hs, hand_specs, canon_lim = ctx["hs"], ctx["hand_specs"], ctx["canon_lim"]
    orig = _read(path)
    own = _limits(ET.fromstring(orig.encode()))
    mine = [r for r in rows if r["side"] in file_sides]
    name = {r["joint"]: hs.vendor_link_prefix.format(side=r["side"]) + hand_specs.strip_side(r["joint"]) for r in mine}
    missing = [name[r["joint"]] for r in mine if name[r["joint"]] not in own]
    if missing:
        die(input_missing(f"{path.name}: joints to clip are missing {missing[:4]} — this file is not this hand's URDF"))
    off = [(name[j], own[name[j]], canon_lim[j]) for j in name if own[name[j]] != canon_lim[j]]
    if off:
        die(
            f"{path.name}: limits differ from the canonical merged URDF {len(off)}/{len(name)} (e.g. {off[0]}) — "
            "it is not the merge source. "
            "Name the source file with --hand-urdf"
        )
    changed = {name[j]: v for j, v in _joint_rows(mine).items() if v[0] != v[1]}
    out = path.with_name(path.stem + SUFFIX + path.suffix)
    new = _with_mark(_rewrite_limits(orig, {j: v[1] for j, v in changed.items()}), ctx["prov"])
    n_attr = _check_only_limits(orig, new, changed, ctx["prov"], path.name)
    got, ref = _meshes(new, out), _meshes(orig, path)
    if [f for _, f in got] != [f for _, f in ref]:
        die(f"{out.name}: meshes do not resolve to the same files as the original")
    n_ok = sum(f is not None for _, f in got)
    print(
        f"[TRUNC] hand URDF {path.name} ({'·'.join(file_sides)}): {len(changed)} joints changed · "
        f"meshes {n_ok}/{len(got)} resolved (same as the original) -> {_rel(out)}"
    )
    return (
        out,
        new,
        {
            "path": _rel(out),
            "source": _rel(path),
            "source_sha256": _sha(path),
            "sides": file_sides,
            "changed_joints": sorted(changed),
            "changed_attrs": n_attr,
            "meshes_resolved": f"{n_ok}/{len(got)}",
        },
    )


def _git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked": bool(g("status", "--porcelain", "--untracked-files=no")),
    }


def _rig_check(rows: list) -> float:
    worst = max(max(abs(r["new_lo"] - r["rig_new_lo"]), abs(r["new_hi"] - r["rig_new_hi"])) for r in rows)
    if worst > RIG_TOL:
        die(
            f"clipped asset limits differ from the IK rig's clipped limits by {worst:.3g} rad (allowed {RIG_TOL}) — "
            "the rig and the asset started from different ranges"
        )
    return worst


def _provenance(rows, a, cdir, tree, win, worst, sides, merged, hands):
    """truncation_provenance.json content: window rule, prior identity, per-joint old/new limits.
    Returns (changed rows, record)."""
    ch = [r for r in rows if (r["lo"], r["hi"]) != (r["new_lo"], r["new_hi"])]
    joint = [
        {
            "joint": r["joint"],
            "side": r["side"],
            "sign": r["sign"],
            "neutral": r["neutral"],
            "old_deg": _deg((r["lo"], r["hi"])),
            "new_deg": _deg((r["new_lo"], r["new_hi"])),
            "old_rad": [r["lo"], r["hi"]],
            "new_rad": [r["new_lo"], r["new_hi"]],
            "ik_rig_new_deg": _deg((r["rig_new_lo"], r["rig_new_hi"])),
        }
        for r in rows
    ]
    prov = {
        "hand": a.hand,
        "combo_dir": cdir,
        "variant_suffix": SUFFIX,
        "tree": tree,
        "generator": "dexengine/robots/urdf/truncate.py",
        "rule": {
            "window": WINDOW,
            "rule_file": _rel(win["rule_file"]),
            "rule_sha256": _sha(pathlib.Path(win["rule_file"])),
            "functions": ["hyperext_clip_targets", "hyperext_clip_range", "build_prior_conf"],
            "prior": _rel(win["prior"]),
            "prior_sha256": win["prior_sha256"],
            "marker": win["marker"],
            "rig_max_abs_diff_rad": worst,
            "rig_tol_rad": RIG_TOL,
        },
        "counts": {
            "considered": len(rows),
            "changed": len(ch),
            "changed_by_side": {s: sum(r["side"] == s for r in ch) for s in sides},
        },
        "merged": merged,
        "hand_urdfs": hands,
        "joints": joint,
    }
    return ch, prov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True, help="hand id (e.g. dg5f_s, wuji2)")
    ap.add_argument(
        "--hand-urdf",
        default=None,
        help="hand URDF (the merge source); with `{side}`, one per side. "
        "Default = HandSpec.vendor_urdf (place = the paths.urdf_mesh rule)",
    )
    a = ap.parse_args()
    hs = hand_specs.HAND_REGISTRY.get(a.hand)
    if hs is None:
        die(f"unregistered hand {a.hand!r}")
    cdir = P.combo_dirname(P.combo_for(a.hand))
    canon = FULL_ROBOT_URDF_DIR / cdir / f"{cdir}.urdf"
    out = FULL_ROBOT_URDF_DIR / f"{cdir}{SUFFIX}" / f"{cdir}{SUFFIX}.urdf"
    tree = _git_state()
    print(
        f"[TREE] {REPO} commit={tree['commit'][:12]} branch={tree['branch']} dirty={tree['dirty_tracked']} "
        f"data_root={P.main_root()}"
    )
    canon_lim = _limits(ET.parse(canon).getroot())
    win = _window(P.hand_spec_dir(a.hand), canon_lim)
    print(win["marker"])
    rows, sides = win["rows"], win["meta"]["sides"]
    worst = _rig_check(rows)
    prov_rel = _rel(out.parent / PROV_NAME)
    ctx = {"hs": hs, "hand_specs": hand_specs, "canon_lim": canon_lim, "prov": prov_rel}
    built = [_do_merged(canon, out, rows, prov_rel)] + [
        _do_hand(p, s, rows, ctx) for p, s in _hand_files(hs, a.hand_urdf, sides)
    ]
    for path, text, info in built:  # write only after every check — never leave a partial output
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, text)
        if _read(path) != text:  # output-level check: the bytes on disk equal the checked text
            die(f"{path}: written bytes differ from the checked text")
        info["sha256"] = _sha(path)
    merged, hands = built[0][2], [b[2] for b in built[1:]]
    ch, prov = _provenance(rows, a, cdir, tree, win, worst, sides, merged, hands)
    (out.parent / PROV_NAME).write_text(json.dumps(prov, ensure_ascii=False, indent=1) + "\n")
    print(
        f"[TRUNC] hand={a.hand} changed={len(ch)}/{len(rows)} merged={merged['path']} "
        f"hand_urdfs={[h['path'] for h in hands]} rig_max_diff={worst:.2g}rad -> {prov_rel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
