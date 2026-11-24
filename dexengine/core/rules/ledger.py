"""Final-ledger naming: ledger axes and keys, ledger files per hand, test-tag classes, and the stage-1 exclusion
lists (clips dropped from retargeting input)."""

from __future__ import annotations

import pathlib
import re

from dexengine.core.errors import InputMissingError
from dexengine.core.rules.identity import active_hand
from dexengine.core.rules.profiles import ALL, COMBO_VARIANTS, DATASET_PROFILE
from dexengine.core.rules.source import _paths

LEDGER_PREFIX = "status_final_"


def ledger_file(axis: str, hand: str, root=None):
    """Final ledger of axis `axis` for hand `hand` — `overfit/hands/<hand>/ledger/status_final_<axis>.tsv`. **The only
    place that builds the name** (the fold writes through it, ledger_files reads through it). The hand is named by the
    **folder**, not the file name."""
    if not axis or "/" in axis:
        raise ValueError(f"ledger axis {axis!r} — must be non-empty and hold no path separator")
    return _paths().run_path("run_ledger", hand, root=root) / f"{LEDGER_PREFIX}{axis}.tsv"


def ledger_files(pattern: str = "*", hand: str | None = None, root=None) -> list:
    """Final ledger files of the hand — the **single entry** for reading ledgers.
    `hand=None` = the running hand (ACTIVE_HAND). Pass a name only to read another hand's adopted set
    (segment split --adopted-from, campaign report).

    Why one entry: consumers globbing `status_final_*.tsv` on their own would **mix two hands' results**. Ledgers
    live inside the hand root (`run_path("run_ledger", hand)`), so another hand's files are in **another folder**
    and a glob cannot reach them.
    `root` = the output checkout (only when a worktree reads the canonical checkout's ledgers — same meaning as
    `dexengine.core.paths.main_root()`). Default: this file's tree.
    """
    h = hand or active_hand()
    d = _paths().run_path("run_ledger", h, root=root)
    if not d.is_dir():
        return []
    return sorted(d.glob(f"{LEDGER_PREFIX}{pattern}.tsv"))


#: Measurements that state a judge chunk's **identity (combo)** — only what the run itself wrote (not human notes,
#: folder names or dates). (file, regex — first group = combo). `chunk_combo` strips the variant suffix
#: (`_aerial`, `_artic`) with dexengine.core.paths.split_combo.
CHUNK_IDENTITY_SOURCES = (
    ("run.env", r"^combo=(\S+)"),  # run environment written by stage2 (with variant)
    ("summary.txt", r"^\[stage2\] \S+ combo=(\S+)"),  # stage2 summary (with variant), also in pre-p0 chunks
    ("run.log", r"^\[stage2\] tag=\S+ combo=(\S+)"),  # stage2 entry marker (with variant)
    ("run.log", r"^\[pipeline\] .*\bcombo=(\S+)"),  # chain entry marker (no variant)
)


def chunk_combo(chunk_dir) -> str:
    """**Base combo** of a judge chunk folder (`<arm>_<hand>` without the variant suffix) — the single owner of chunk
    identity.

    The only evidence is what the run wrote: the judge-plan name `p0/plans2_<combo>.jsonl` + `CHUNK_IDENTITY_SOURCES`.
    · Evidence that **disagrees** → ValueError (two identities in one chunk — a mixed folder).
    · **No evidence** → "" — the caller fails loudly (an unknown input is never silently attached to a hand).
    All evidence is read — chunks without p0 or run.log have only the `combo=` of summary.txt.
    An unregistered arm or hand name is returned as is — it cannot equal a current combo, so it counts as another
    hand."""
    import re as _re
    from pathlib import Path as _P

    d = _P(chunk_dir)
    seen: dict[str, str] = {}
    for f in sorted((d / "p0").glob("plans2_*.jsonl")) if (d / "p0").is_dir() else []:
        seen.setdefault(f.stem[len("plans2_") :], f"p0/{f.name}")
    for name, rx in CHUNK_IDENTITY_SOURCES:
        f = d / name
        if not f.is_file():
            continue
        for m in _re.finditer(rx, f.read_text(errors="replace"), _re.M):
            seen.setdefault(m.group(1), name)
    P_ = _paths()
    base: dict[str, str] = {}
    for c, src in seen.items():
        arm, hand, _v = P_.split_combo(c, P_.ARM_IDS, P_.HAND_IDS, unsplit_ok=True)  # unregistered = as is
        b = P_.combo_id(arm, hand) if arm and hand else c
        base.setdefault(b, src)
    if len(base) > 1:
        raise ValueError(f"chunk {d.name} has several identities: " + " · ".join(f"{b}({s})" for b, s in base.items()))
    return next(iter(base), "")


def ledger_files_for_hand(pattern: str, hand: str) -> list:
    """Alias of `ledger_files(pattern, hand)` — there is one entry."""
    return ledger_files(pattern, hand)


def ledger_axis(path) -> str:
    """Take the **axis name** out of a ledger file name (`status_final_<axis>.tsv`)."""
    from pathlib import Path as _P

    stem = _P(path).stem
    if not stem.startswith(LEDGER_PREFIX):
        raise ValueError(f"ledger name outside the convention: {_P(path).name} — `{LEDGER_PREFIX}<axis>.tsv`")
    return stem[len(LEDGER_PREFIX) :]


#: Test-tag prefixes — tag classes **excluded** from the canonical fold. The canonical-50 regression and the canary
#: write judge/bake output to the same `overfit/stage2/<tag>` as the canon but are not campaign results (mixed in, a
#: regression re-run would become the fold's "latest ledger"). Owned here only — the shell uses
#: `--test-tag-prefix <kind>`.
#: `regress_retry` = the D37 re-judge that the canonical-50 regression reproduces. Its chunk sits **inside** the first
#: judge chunk, so top-level chunk scans (retry runs, budget audit, regression list) do not read it as a campaign
#: retry or a second regression.
TEST_TAG_PREFIXES = {"regress": "G1regr_", "regress_retry": "G1regrr_", "canary": "G1k_"}


def test_tag_kind(tag: str) -> str:
    """The kind (regress · regress_retry · canary) when the tag is a test class, else an empty string. Longer
    prefixes are matched first (G1regrr_ starts with G1regr_, so the short one winning would read a retry chunk as a
    regression). stage2 reads it to enforce test-tag isolation."""
    for kind, pfx in sorted(TEST_TAG_PREFIXES.items(), key=lambda kv: -len(kv[1])):
        if tag.startswith(pfx):
            return kind
    return ""


#: ── Source-defect clip ledger ─────────────
#: Line form `<clip><TAB><reason>[<TAB><hand scope>]`. One reader, `retarget_excluded`, and one owner per kind of line:
#:   human decisions   the repository's `configs/datasets/<ds>/excluded_clips.txt` (REPO_EXCLUDED_REL), so a data root
#:                     built from the raw datasets applies them too. The data root's ledger may still carry copies of
#:                     them (from an earlier build): they are only checked — a human line there whose clip the
#:                     repository does not exclude dies. A dataset without a repository file keeps its data-root human
#:                     lines as its statement until they move into the repository.
#:   generator lines   `<source>:<word> — <description>` (retarget_excluded_record) — the data root's
#:                     `precompute/<ds>/diag/excluded_clips.txt`, written by the pipeline.
#: `dexengine/placement/ledger.py` uses it for the `SOURCE_DEFECT` verdict; the shell (stage-1 count check) reads
#: through it.
#:
#: Note — this list and "clips with no stage-1 output" are **not the same set**: of the 4 taco clips, 1
#: (`brush_brush_bowl__20231005_188`) fails retargeting and the other 3 produce output but are excluded at the judge
#: layer. So the count check is the rule **"every clip without output must be explained by this ledger"**, not a
#: subtraction of the list.
RETARGET_EXCLUDED_REL = "precompute/{ds}/diag/excluded_clips.txt"
#: A generator line's reason (`<source>:<word> …`, retarget_excluded_record) — the other lines are human decisions.
GENERATOR_LINE = re.compile(r"[^\s:]+:\S")
REPO_EXCLUDED_REL = "datasets/{ds}/excluded_clips.txt"  # under configs/


def repo_excluded_path(dataset: str):
    """The repository's human exclusion decisions of `dataset` (may not exist)."""
    from dexengine.core.repo import CONFIGS_DIR  # noqa: PLC0415 — standard-library-only module

    retarget_excluded_path(dataset)  # an unknown dataset dies there
    return CONFIGS_DIR / REPO_EXCLUDED_REL.format(ds=dataset)


def retarget_excluded_path(dataset: str):
    if dataset not in ALL:
        raise SystemExit(f"[rules] unknown dataset '{dataset}' — the list is owned by dexengine.core.datasets.ALL")
    return _paths().main_root() / RETARGET_EXCLUDED_REL.format(ds=dataset)  # precompute/ = data root


#: Hand scope (third, optional column) — `<clip><TAB><reason><TAB><hand scope>`. Hand scope = `*` or a comma list
#: of hand ids (dexengine.core.paths.HAND_IDS). No column or `*` = every hand. A hand-scoped line applies only when
#: running that hand, so one hand's decision never drops the clip from another hand's roster. With several lines
#: for one clip, the first wins (human lines come before generator lines).
EXCLUDE_ALL_HANDS = "*"


def excluded_scope(line: str) -> tuple[str, ...] | None:
    """Hand scope of one ledger line — None = every hand, tuple = those hands. A retired hand id
    (locations.hands.RETIRED_HAND_IDS) is dropped from the tuple (the line then applies to fewer hands, never more);
    any other unknown hand id or an empty list raises (never silently widened)."""
    parts = line.rstrip("\n").split("\t")
    scope = parts[2].strip() if len(parts) > 2 else ""
    if scope in ("", EXCLUDE_ALL_HANDS):
        return None
    hands = tuple(h.strip() for h in scope.split(","))
    known = set(_paths().HAND_IDS)
    if not all(hands) or not set(hands) <= known | set(_paths().RETIRED_HAND_IDS):
        raise SystemExit(
            f"[rules] exclusion-ledger hand scope {scope!r} — '*' or a comma list of hand ids "
            f"(registered: {sorted(known)})"
        )
    return tuple(h for h in hands if h in known)


def excluded_from_file(p, hand: str | None = None) -> dict[str, str]:
    """The **only parser** of one ledger file — {clip: reason}. A hand-scoped line is returned only when `hand` (None
    = active_hand() once such a line is met — raises without DEXENGINE_HAND) is in scope. A line without a reason
    raises."""
    out: dict[str, str] = {}
    for ln in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        clip, _, rest = ln.partition("\t")
        clip, why = clip.strip(), rest.partition("\t")[0].strip()
        if not clip:
            continue
        if not why:
            # A line without a reason is a **failure** — an exclusion is a decision, and a decision carries a reason.
            raise InputMissingError(
                f"[rules] {p}: '{clip}' has no reason — form: <clip><TAB><reason>[<TAB><hand scope>]"
            )
        scope = excluded_scope(ln)
        if scope is not None:
            hand = hand or active_hand()
            if hand not in scope:
                continue
        out.setdefault(clip, why)
    return out


def retarget_excluded(dataset: str, hand: str | None = None) -> dict[str, str]:
    """Source-defect clips of the dataset {clip: reason}: the repository's human decisions and the data root's
    generator lines (section comment; hand-scoped lines only for that hand — excluded_from_file). An **unknown dataset
    raises**, and so does a data-root human line the repository statement lacks. Missing files give an empty list —
    not a silent pass, because the consumer applies the rule "an unexplained gap is a failure"."""
    p = retarget_excluded_path(dataset)
    site = excluded_from_file(p, hand) if p.is_file() else {}
    r = repo_excluded_path(dataset)
    if not r.is_file():
        return site
    out = excluded_from_file(r, hand)
    if stray := sorted(c for c, why in site.items() if not GENERATOR_LINE.match(why) and c not in out):
        raise InputMissingError(
            f"[rules] {p}: human exclusion line(s) for {stray[:3]} (total {len(stray)}) are not in the repository "
            f"statement {r}, which owns human decisions — add them there or remove them here"
        )
    for clip, why in site.items():
        if GENERATOR_LINE.match(why):
            out.setdefault(clip, why)
    return out


def retarget_excluded_by(dataset: str, source: str) -> dict[str, str]:
    """Exclusion lines written by generator `source` {clip: reason word} — the reader of the line form
    `<clip><TAB><source>:<word> — <description>` (paired with the writer retarget_excluded_record; the form is not
    split anywhere else). Human lines and other generators' lines are not returned. The generator owns the
    vocabulary."""
    tag, out = f"{source}:", {}
    for clip, why in retarget_excluded(dataset).items():
        if why.startswith(tag):
            word = (why[len(tag) :].split() or [""])[0]
            if not word:
                raise InputMissingError(
                    f"[rules] {retarget_excluded_path(dataset)}: the {tag} line of '{clip}' has no reason word"
                )
            out[clip] = word
    return out


def retarget_excluded_record(
    dataset: str, source: str, verdicts: dict[str, tuple[str, str]], scanned, hand: str | None = None
) -> dict:
    """Write the exclusions {clip: (reason word, description)} judged by generator `source` (e.g. `D1g`) into the
    dataset's exclusion ledger — the ledger's **write entry** (same file as the readers retarget_excluded and
    retarget_excluded_by). Line form = `<source>:<word> — <description>` (no spaces in the word). Rule: lines that
    `source` wrote (reason starts with `<source>:`) for clips scanned this time (`scanned`) are replaced by this
    verdict — a clip that now converts loses its line (no stale exclusion stays). Human lines, other generators'
    lines and lines of unscanned clips stay. When a human line already excludes the clip it stays and no generator
    line is added (one line per clip). A verdict on a clip outside `scanned` raises. Atomic write, skipped when the
    content is unchanged. Returns {path, written, n_source, n_kept_other}.
    `hand` (a hand id) writes hand-scoped lines (third column) for a verdict that depends on the robot hand: only this
    generator's lines of that one hand are replaced, and a clip another line already claims for every hand or for
    this hand gets no second line."""
    p = retarget_excluded_path(dataset)
    scanned, tag = set(scanned), f"{source}:"
    if hand is not None and hand not in _paths().HAND_IDS:
        raise SystemExit(f"[rules] exclusion-ledger hand scope {hand!r} — not a registered hand id")
    mine = None if hand is None else (hand,)
    if not source or ":" in source or "\t" in source:
        raise SystemExit(f"[rules] exclusion-ledger generator name {source!r} — must be non-empty, no ':' or tab")
    if stray := sorted(set(verdicts) - scanned):
        raise SystemExit(
            f"[rules] exclusion verdicts name clips not scanned this time {stray[:3]} — write scanned clips only"
        )
    old = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    keep, taken = [], set()
    for ln in old:
        clip, _, why = ln.partition("\t")
        body = bool(ln.strip()) and not ln.lstrip().startswith("#")
        scope = excluded_scope(ln) if body else None
        if body and why.strip().startswith(tag) and clip.strip() in scanned and scope == mine:
            continue  # a line this generator wrote before (same hand scope) — replaced by this verdict
        keep.append(ln)
        if body and (scope is None or (hand is not None and hand in scope)):  # the line claims the clip for this run
            taken.add(clip.strip())
    if bad := sorted(c for c, (w, _y) in verdicts.items() if not w or len(str(w).split()) != 1):
        raise SystemExit(f"[rules] exclusion reason word empty or holding spaces {bad[:3]} — must be one word")
    add = [
        f"{c}\t{tag}{w}"
        + (f" — {' '.join(str(y).split())}" if str(y).strip() else "")
        + ("" if hand is None else f"\t{hand}")
        for c, (w, y) in sorted(verdicts.items())
        if c not in taken
    ]
    text = "".join(x + "\n" for x in keep + add)
    wrote = text != ("".join(x + "\n" for x in old))
    if wrote:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)
    return {"path": p, "written": wrote, "n_source": len(add), "n_kept_other": len(verdicts) - len(add)}


def ledger_key(ds: str, emb: str = "bimanual", variant: str = "") -> str:
    """The **single rule** for the final ledger axis name (`overfit/stage2/G1/status_final_<key>.tsv`).

    key = <dataset>[-<emb>][-<variant>]. The variant is held by the dataset profile (`_artic` → `-artic`).
    Why it is needed: name a re-judge chunk tag `<prefix>_<dataset>_NN` and the results of a
    variant axis (arctic-artic·show3d-aerial) are **not caught** by the ledger fold (`G1*_<key>_NN` pattern) —
    they are silently discarded. Both the place that makes the tag and the place that folds read this one function.
    """
    if ds not in DATASET_PROFILE:  # an unknown input is never let through silently
        raise KeyError(f"unknown dataset: {ds!r} (known: {', '.join(sorted(DATASET_PROFILE))})")
    pr = DATASET_PROFILE[ds]
    key = ds + ("" if emb in ("", "bimanual") else f"-{emb}")
    # The variant **the run actually used** wins — when a run of a dataset without a profile variant ran with one, the
    # axis name must carry it for the results to fold into the ledger. The canonical path has no such run now (desk
    # profiles always use the desk scene, D45; the D20 aerial scene only applies to desk_mode none profiles), but
    # ledger names like `hot3d-aerial` must stay readable. When not given, the profile value is used.
    v = variant or pr.variant
    return key + (v.replace("_", "-", 1) if v else "")


def ledger_axis_parts(key: str) -> tuple[str, str, str]:
    """Inverse of `ledger_key` — axis key → (dataset, emb, variant). Found by **trying every input of the forward
    function** rather than restating the rule (a key-rule change needs no edit here). Raises when not found (no
    silent guess)."""
    for ds in DATASET_PROFILE:
        for emb in ("bimanual", "right", "left"):
            for v in COMBO_VARIANTS:
                try:
                    if ledger_key(ds, emb, v) == key:
                        return ds, emb, v
                except KeyError:  # noqa: PERF203 — an input the forward rule refuses is simply not a match
                    continue
    raise KeyError(f"no (dataset, emb, variant) produces ledger axis {key!r}")


def ledger_axis_variant(key: str) -> str:
    """The variant **the axis key actually used** (`_aerial`, `_artic`, or "") — including the profile's variant.

    Why: a substring test such as `"aerial" in key or "artic" in key` is a lookup table of yesterday's data, not a
    rule — it silently breaks when a variant is added or a dataset name contains those letters. The answer here
    inverts the forward function `ledger_key` (`ledger_axis_parts`) and adds the profile variant; an unknown axis
    raises.
    """
    ds, _emb, v = ledger_axis_parts(key)
    return v or DATASET_PROFILE[ds].variant
