"""The training generation of a campaign checkpoint's clips: the per-clip checks behind the [CANON_USE] marker.

A campaign cell's checkpoint was trained on one tree fingerprint per clip (sha256 of <tree>/task_info.json). The queue
root (env DEXENGINE_RL_QUEUE_ROOT) declares what happened to each clip's tree since:
  frozen table   <queue root>/clip_tree_sha/<hand>/<manifest sha12>.tsv: the training fingerprint of every primary
                 clip of a manifest (trained_of; it must reproduce the manifest's per-axis fingerprints)
  TURNOVER.tsv   declarations per (hand, axis, clip): turnover (an old generation) · same_gen (the generation goes on
                 at a new fingerprint) · new · scene · decide · withdraw (read_turnover · turns_of)
  tree_content   content snapshots declared before a tree is overwritten (read_snaps)
slot_state turns those into one word per (checkpoint slot, clip): current · pending · turned, or raises when a tree
changed under a trained slot without a declaration.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path

from dexengine.core import paths as P

# ── slot rules ───────────────────────────────────────────────────────────────────────────────────────────────────────
#: TURNOVER row kinds. scene = the tree is unchanged but the clip's training scene changed (only the lineage of the
#: last scene row stays current) · decide = a recorded decision retrains a canon clip as it is (turns no slot) ·
#: withdraw = a later row taking back a decide (same gen · from_sha).
SCENE = "scene"
DECIDE = "decide"
WITHDRAW = "withdraw"
TURN_KINDS = ("turnover", "same_gen", "new", SCENE, DECIDE, WITHDRAW)
#: The lineage a caller passes when it asks about the generation of a tree, not about one slot (no scene row turns a
#: slot for it).
ANY_LINEAGE = "*"


def to_sha_problem(kind: str, frm: str, to: str, is_sha) -> str:
    """What a TURNOVER row's to_sha breaks for its kind ('' = valid · turnover_problems — is_sha its 64-hex
    test): turnover '-' · same_gen 64 hex ≠ from_sha · new 64 hex · scene · decide = from_sha (tree as is). A kind
    outside TURN_KINDS raises (no kind passes unchecked)."""
    if kind == "turnover":
        return "" if to == "-" else "turnover: to_sha is '-'"
    if kind == "same_gen":
        return "" if is_sha(to) and to != frm else "same_gen: to_sha = 64 hex · differs from from_sha"
    if kind == "new":
        return "" if is_sha(to) else "new: to_sha = 64 hex"
    if kind == SCENE:
        return "" if to == frm else "scene: to_sha = from_sha (tree unchanged · only the training scene changes)"
    if kind in (DECIDE, WITHDRAW):
        return "" if to == frm else f"{kind}: to_sha = from_sha (tree · scene unchanged)"
    raise ValueError(f"to_sha_problem: kind {kind!r} ∉ {TURN_KINDS}")


def slot_gen(trained: str, turns: list[dict]) -> frozenset[str]:
    """A slot's generation — its training fingerprint plus every same_gen to_sha reachable from it (turns = the TURNOVER
    rows of the slot's (axis, clip)). A declaration is matched to a slot through this, never through the frozen
    fingerprint alone."""
    gen, todo = {trained}, [trained]
    while todo:
        x = todo.pop()
        for t in turns:
            if t["kind"] == "same_gen" and t["from_sha"] == x and t["to_sha"] not in gen:
                gen.add(t["to_sha"])
                todo.append(t["to_sha"])
    return frozenset(gen)


def standing(turns: list[dict]) -> list[dict]:
    """The clip's TURNOVER rows still in force (turns = its rows in ledger order — turns_of): every row but
    the turnover rows a LATER same_gen of the same from_sha withdrew (an earlier same_gen or one of another
    fingerprint withdraws nothing) and the decides a LATER withdraw of the same gen · from_sha took back."""
    last = {t["from_sha"]: i for i, t in enumerate(turns) if t["kind"] == "same_gen"}
    wd = {(t["gen"], t["from_sha"]): i for i, t in enumerate(turns) if t["kind"] == WITHDRAW}
    return [
        t
        for i, t in enumerate(turns)
        if not (t["kind"] == "turnover" and last.get(t["from_sha"], -1) > i)
        and not (t["kind"] == DECIDE and wd.get((t["gen"], t["from_sha"]), -1) > i)
    ]


def slot_state(
    trained: str, current: str | None, turns: list[dict], snaps=frozenset(), lineage: str | None = None
) -> str:
    """The one predicate of a slot (bundle, clip) — its training fingerprint against the clip's tree now, the clip's
    TURNOVER rows and its content snapshots (snaps = the tree fingerprints the clip's snapshot rows declared before an
    overwrite — read_snaps): a standing turnover row (`standing` — not withdrawn by a later same_gen of the
    same from_sha) whose from_sha is in the slot's generation (slot_gen) → 'turned'
    (an old generation — history only) · the current fingerprint in it → 'current' · a snapshot of the generation and
    the tree moved on (or gone) → 'pending' (declared rebuild waiting for its split result — the old checkpoint stays
    the owner until same_gen or turnover lands) · anything else raises ValueError: the tree changed under a trained slot
    with nobody declaring it (never silently pending). current None = the clip has no tree now: allowed only for a
    turned or pending slot — else ValueError. A scene row (the comment above TURN_KINDS) of the slot's generation
    makes every slot whose `lineage` (lineage_of of its manifest) is not the LAST such row's gen 'turned';
    lineage None with such a row raises ValueError (a caller that does not say which slot it asks about) · ANY_LINEAGE
    = the generation question (no slot is turned by a scene row)."""
    gen = slot_gen(trained, turns)
    if any(t["kind"] == "turnover" and t["from_sha"] in gen for t in standing(turns)):
        return "turned"
    scene = [t for t in turns if t["kind"] == SCENE and t["from_sha"] in gen]
    if scene and lineage is None:
        raise ValueError(
            f"slot {trained[:12]} of a clip with a scene turnover row (row {scene[-1].get('n', '?')} · "
            f"lineage {scene[-1]['gen']}) — deciding needs the slot's lineage (the caller passes it)"
        )
    if scene and lineage != ANY_LINEAGE and lineage != scene[-1]["gen"]:
        return "turned"
    if current not in gen and gen & set(snaps):
        return "pending"
    if current is None:
        raise ValueError(
            f"no tree — no TURNOVER turnover declaration covers the slot with training fingerprint "
            f"{trained[:12]} (a clip whose name vanished in a generation change declares turnover first)"
        )
    if current in gen:
        return "current"
    raise ValueError(
        f"undeclared generation change — training fingerprint {trained[:12]} ≠ current tree "
        f"{current[:12]} and the clip has no TURNOVER declaration (declare "
        "turnover | same_gen)"
    )


# ── withdraw rows ──────────────────────────────────────────────────────────────────────────────────────────────
def withdraw_keys(r: dict, seen: set, bad: list) -> list[tuple]:
    """parse_turnover's uniqueness keys of a withdraw row (one per withdrawn decide) — and its problems into `bad`:
    no standing decide of the same (hand, axis, clip, gen, from_sha) before it (the decide's own key in `seen`) · a
    why that does not name the row (row <n>)."""
    if ("from", r["hand"], r["axis"], r["clip"], DECIDE, r["from_sha"], r["gen"]) not in seen:
        bad.append(f"no earlier decide row to withdraw ({r['gen']} {r['axis']}:{r['clip']} {r['from_sha'][:12]})")
    if not re.search(r"\brow \d+", r["why"]):
        bad.append("withdraw: why names the withdrawn row number (row <n>)")
    return [("withdraw", r["hand"], r["axis"], r["clip"], r["gen"], r["from_sha"])]


# ── frozen training fingerprints · TURNOVER.tsv ──────────────────────────────────────────────────────────────────
#: The frozen training-fingerprint tables under the queue root, and their head.
FROZEN_DIR = "clip_tree_sha"
FROZEN_HEAD = ("manifest_sha", "axis", "clip", "sha256")
HEX64 = frozenset("0123456789abcdef")


def is_sha(x: str) -> bool:
    """64 lowercase hex characters."""
    return len(x) == 64 and set(x) <= HEX64


def axis_fp(pairs) -> str:
    """The axis tree fingerprint: sha256 over `clip sha` lines in sorted clip order (pairs = (clip, sha))."""
    return hashlib.sha256("".join(f"{c} {s}\n" for c, s in sorted(pairs)).encode()).hexdigest()


def tree_sha(root: Path, clip: str) -> str:
    """sha256 of <root>/<tree name>/task_info.json — the clip's fingerprint at that tree root."""
    f = Path(root) / P.tree_name(clip) / "task_info.json"
    if not f.is_file():
        raise ValueError(f"{f}: tree task_info.json missing — the clip's current generation cannot be measured")
    return hashlib.sha256(f.read_bytes()).hexdigest()


def primaries(man: dict) -> dict[str, list[str]]:
    """{axis: its primary clips} of a manifest (a dup is the primary of another bundle of the same axis)."""
    out: dict[str, list[str]] = {}
    for b in man["bundles"]:
        out.setdefault(b["axis"], []).extend(c for c in b["clips"] if c not in b["dups"])
    return out


def reproduce(man: dict, shas: dict[tuple[str, str], str]) -> list[str]:
    """Axes whose fingerprint the per-clip shas do not reproduce, and clip-set differences — [] = the shas are the
    manifest's (each primary once, nothing else)."""
    prim, bad = primaries(man), []
    want = {(ax, c) for ax, cs in prim.items() for c in cs}
    if set(shas) != want:
        bad.append(
            f"clip set ≠ manifest main clips: missing {len(want - set(shas))} {sorted(want - set(shas))[:2]} · "
            f"outside {len(set(shas) - want)} {sorted(set(shas) - want)[:2]}"
        )
    for a in man["axes"]:
        cs = prim.get(a["axis"], [])
        if not cs:
            continue
        got = axis_fp((c, shas.get((a["axis"], c), "")) for c in cs)
        if got != a.get("tree_sha256"):
            bad.append(f"{a['axis']}: axis fingerprint {got[:12]} ≠ manifest {str(a.get('tree_sha256'))[:12]}")
    return bad


def read_table(path: Path) -> list[tuple[str, str, str, str]]:
    """The rows of a frozen training-fingerprint table (head FROZEN_HEAD; a bad row raises ValueError)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != FROZEN_HEAD:
        raise ValueError(f"{path}: header ≠ {FROZEN_HEAD}")
    out = []
    for ln in lines[1:]:
        c = tuple(ln.split("\t"))
        if len(c) != len(FROZEN_HEAD) or not is_sha(c[0]) or not is_sha(c[3]):
            raise ValueError(f"{path}: row {ln[:90]!r} — needs {len(FROZEN_HEAD)} fields · fingerprint·sha 64 hex")
        out.append(c)
    return out


def frozen_path(qroot: Path, hand: str, man_sha: str) -> Path:
    """The frozen table of a manifest under the queue root."""
    return Path(qroot) / FROZEN_DIR / hand / f"{man_sha[:12]}.tsv"


def check_table(man: dict, man_sha: str, rows: list[tuple]) -> dict[tuple[str, str], str]:
    """{(axis, clip): sha} of a frozen table that is this manifest's (manifest_sha column · per-axis re-hash · clip
    set) — else ValueError."""
    fps = {r[0] for r in rows}
    if fps != {man_sha}:
        raise ValueError(f"frozen table manifest sha {sorted(x[:12] for x in fps)} ≠ {man_sha[:12]}")
    shas = {(r[1], r[2]): r[3] for r in rows}
    if len(shas) != len(rows):
        raise ValueError(f"frozen table has an (axis, clip) twice — rows {len(rows)} · unique {len(shas)}")
    bad = reproduce(man, shas)
    if bad:
        raise ValueError(
            f"frozen table does not reproduce the axis fingerprints of manifest {man['fingerprint'][:12]}: {bad[:3]}"
        )
    return shas


def trained_of(qroot: Path, man: dict, man_sha: str) -> dict[tuple[str, str], str]:
    """{(axis, clip): training fingerprint} of a manifest's bundles from the queue root's frozen table (man_sha = the
    manifest file's sha256); a ValueError when the queue root has no table for it."""
    f = frozen_path(qroot, man["hand"], man_sha)
    if not f.is_file():
        raise ValueError(
            f"{man['hand']} manifest {man_sha[:12]} (fingerprint {man['fingerprint'][:12]}): per-clip "
            f"training-fingerprint frozen table {f} missing — no canonical pick without training "
            "fingerprints"
        )
    return check_table(man, man_sha, read_table(f))


def load_manifest(path: Path) -> tuple[dict, str]:
    """(manifest, its file sha256)."""
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def lineage_of(man: dict) -> str:
    """A manifest's lineage: its `lineage.gen` (a partial retrain manifest), or "main" for a manifest without the
    record."""
    lin = man.get("lineage")
    if lin is None:
        return "main"
    if not isinstance(lin, dict) or not lin.get("gen") or lin["gen"] == "main":
        raise ValueError(f"manifest lineage record {lin!r} — needs a gen (not main)")
    return lin["gen"]


#: The generation declarations of the queue root, and their head.
TURNOVER_TSV = "TURNOVER.tsv"
TURNOVER_HEAD = ("at", "hand", "axis", "clip", "kind", "gen", "from_sha", "to_sha", "who", "why", "source")


def turnover_problems(r: dict) -> list[str]:
    """What one TURNOVER row breaks ([] = valid): kind ∈ TURN_KINDS · from_sha 64 hex ('-' = new) · to_sha
    by kind (to_sha_problem) · no empty field (a decide row states its reason in why) · at ISO · source
    64 hex (the input file)."""
    p = []
    if r["kind"] not in TURN_KINDS:
        p.append(f"kind {r['kind']!r} ∉ {TURN_KINDS}")
    if not (r["from_sha"] == "-" if r["kind"] == "new" else is_sha(r["from_sha"])) or not is_sha(r["source"]):
        p.append("from_sha('-' = new) · source = 64 hex")
    if r["kind"] in TURN_KINDS and (x := to_sha_problem(r["kind"], r["from_sha"], r["to_sha"], is_sha)):
        p.append(x)
    if not all(r[k].strip() for k in ("hand", "axis", "clip", "gen", "who", "why")) or r["gen"] == "main":
        p.append("hand · axis · clip · gen (not main) · who · why not empty")
    if P.clip_name(r["clip"]) != r["clip"]:
        p.append(f"clip {r['clip']!r} is a clip name (no {P.TREE_SFX})")
    try:
        datetime.datetime.fromisoformat(r["at"])
    except ValueError:
        p.append(f"at {r['at']!r} is an ISO time")
    return p


def read_turnover(qroot: Path) -> list[dict]:
    """TURNOVER.tsv rows of the queue root (parse_turnover); the file must exist."""
    f = Path(qroot) / TURNOVER_TSV
    if not f.is_file():
        raise ValueError(f"{f} missing — no canonical pick without the generation-change ledger")
    return parse_turnover(f)


def parse_turnover(f: Path) -> list[dict]:
    """A TURNOVER file's rows in file order, each with its 1-based row number `n`: a bad row, the same (hand, gen,
    axis, clip) twice or the same (hand, axis, clip, kind, from_sha) twice raises ValueError. A scene row's from-key
    carries its gen (SCENE: one tree, many lineages) · withdraw rows: withdraw_keys."""
    lines = f.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != TURNOVER_HEAD:
        raise ValueError(f"{f}: header ≠ {TURNOVER_HEAD}")
    out, seen = [], set()
    for i, ln in enumerate(lines[1:], 1):
        c = ln.split("\t")
        if len(c) != len(TURNOVER_HEAD):
            raise ValueError(f"{f}: row {i} fields {len(c)} ≠ {len(TURNOVER_HEAD)}")
        r = dict(zip(TURNOVER_HEAD, c, strict=True), n=i)
        bad = turnover_problems(r)
        keys = (
            withdraw_keys(r, seen, bad)
            if r["kind"] == WITHDRAW
            else [
                ("gen", r["hand"], r["gen"], r["axis"], r["clip"]),
                ("from", r["hand"], r["axis"], r["clip"], r["kind"], r["from_sha"])
                + ((r["gen"],) if r["kind"] in (SCENE, DECIDE) else ()),
            ]
        )
        bad += [f"the same {k[0]} twice" for k in keys if k in seen]
        if bad:
            raise ValueError(f"{f}: row {i} — {bad}")
        seen.update(keys)
        out.append(r)
    return out


def turns_of(rows: list[dict], hand: str) -> dict[tuple[str, str], list[dict]]:
    """{(axis, clip): [rows]} of one hand."""
    out: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["hand"] == hand:
            out.setdefault((r["axis"], r["clip"]), []).append(r)
    return out


# ── content snapshots ────────────────────────────────────────────────────────────────────────────────────────────
#: Content snapshots under the queue root (per hand; *.split.tsv files are split results, not snapshots).
SNAPSHOT_DIR = "tree_content"
SNAP_HEAD = ("axis", "clip", "tree_sha", "window", "parts")
SPLIT_SFX = ".split.tsv"
_HEX = frozenset("0123456789abcdef")


def _is_sha(x: str) -> bool:
    return len(x) == 64 and set(x) <= _HEX


def _rows(f: Path, head: tuple) -> list[dict]:
    lines = f.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != head:
        raise ValueError(f"{f}: header ≠ {head}")
    out = [dict(zip(head, ln.split("\t"), strict=False)) for ln in lines[1:] if ln]
    bad = [r for r in out if len(r) != len(head) or not _is_sha(r.get("from_sha", r.get("tree_sha", "")))]
    if bad:
        raise ValueError(f"{f}: rows with missing fields or a fingerprint that is not 64 hex: {len(bad)}")
    return out


def read_snaps(qroot: Path, hand: str) -> dict[tuple[str, str], frozenset]:
    """{(axis, clip): the tree fingerprints its content snapshots declared} — every snapshot of the hand."""
    out: dict[tuple[str, str], set] = {}
    for f in sorted((Path(qroot) / SNAPSHOT_DIR / hand).glob("*.tsv")):
        if not f.name.endswith(SPLIT_SFX):
            for r in _rows(f, SNAP_HEAD):
                out.setdefault((r["axis"], r["clip"]), set()).add(r["tree_sha"])
    return {k: frozenset(v) for k, v in out.items()}
