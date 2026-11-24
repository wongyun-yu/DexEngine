#!/usr/bin/env python3
"""E5 manifest text fill — a converter manifest (manifest.py, the formal smallset_train.tsv) → the same rows with
task_text = the clip's per-episode text from the final root (meta/episodes.jsonl `tasks[0]` — text v1.1, loaded by
the record text store) and lang_source = --lang-source (default episode_v11).

Why a separate tool: the converter stays unchanged — the only change of the formal build is the manifest's task_text
column. GR00T reads the text (episode task); DP ignores it (axis one-hot).

  python -m experiments.policies.data.manifest_text --manifest <in.tsv> --roots orig=<root>,aug=<root> --combo
  rby1_wuji2 \
      --out-dir <dir> [--table <texts.jsonl>] [--lang-source episode_v11]

Episode lookup = the converter's own key (clip, round, look_name(e)) in
<root>/data/<ds>/collect/<combo>/meta/episodes.jsonl (look_name is imported from convert.py — one owner).
A row is COVERED when the episode record has text.instruction (non-empty) · tasks[0] == text.instruction ·
tasks[0] != piece_tasks[0] (the pre-text label, e.g.
's08-phone use 02'). Any row not covered dies after the report is written (the report lists every miss).
--table (optional oracle · measurement vs statement): the v1.1 text table rows of (dataset, base_clip) — counts equal
/ differ / absent; it never changes the output. Position words (count only): rows with a non-empty `variant` (moved
placement) whose text has a position word, raw and with hand phrases ('left hand', 'right hands', 'left-handed', …)
removed first.

--pair <our text manifest> (kinematic pair — the matched kinematic set = the RL reference replayed
without physics, same scenes): instead of reading episode meta, task_text / lang_source come from OUR text-filled
manifest row of the same scene — key (dataset, root without its 'kin_' prefix, clip, round, look), or the row's
pair_root · pair_round when the list carries them (pair_key); a row whose root
lacks 'kin_', a scene absent from ours, or a scene matched twice dies; lang_source = ours + '_pair'. Rows of ours left
unpaired are counted (report).
Outputs (out-dir must not hold them already): manifest_all.tsv · manifest_<axis>.tsv (one per axis, input order)
· text_report.json · texts.sha256. Marker lines [E5_TEXT] … · last line [E5_TEXT] DONE … | [E5_TEXT] FAIL ….
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys

from dexengine.core.hashing import sha256_file
from experiments.policies.data.convert import look_name  # the converter's key rule

TAG = "[E5_TEXT]"
POS_WORDS = (
    "left",
    "right",
    "near",
    "nearer",
    "nearest",
    "nearby",
    "far",
    "farther",
    "farthest",
    "front",
    "behind",
    "beside",
    "next to",
    "close to",
    "closer",
    "center",
    "centre",
    "middle",
    "corner",
    "edge",
)
POS_RE = re.compile(r"\b(" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in POS_WORDS) + r")\b", re.I)
HAND_RE = re.compile(r"\b(left|right)(\s+and\s+(left|right))?[\s-]+hand(s|ed)?\b", re.I)


def die(m: str) -> None:
    print(f"{TAG} FAIL {m}", flush=True)
    sys.exit(1)


def pos_words(text: str, drop_hands: bool) -> list[str]:
    t = HAND_RE.sub(" ", text) if drop_hands else text
    return [m.group(1).lower() for m in POS_RE.finditer(t)]


def pair_key(r: dict) -> tuple:
    """--pair: OUR scene key of a kinematic row — its pair_root · pair_round columns when the list carries them (the
    kin list owner kin_manifest.py writes our root/round on every row: a kinematic episode may sit in another root or
    round than ours, e.g. a re-rendered scene root or an original placement our aug round holds), else the old rule
    (root without kin_, same round). A row with one of the two columns empty dies."""
    pr, pd = r.get("pair_root"), r.get("pair_round")
    if pr is None and pd is None:
        return r["dataset"], r["root"][len("kin_") :], r["clip"], r["round"], r["look"]
    if not pr or not pd:
        die(f"--pair: row {r['clip']} has pair_root {pr!r} · pair_round {pd!r} (both or neither)")
    return r["dataset"], pr, r["clip"], pd, r["look"]


def _episodes_by_key(src):
    """Index the episodes of one collect tree by (clip, round, look); a key matching two episodes dies."""
    by = {}
    with open(os.path.join(src, "meta", "episodes.jsonl")) as fh:
        for ln in fh:
            if ln.strip():
                e = json.loads(ln)
                k = (e["clip"], str(e.get("round")), look_name(e))
                if k in by:
                    die(f"{src}: key {k} matches two episodes")
                by[k] = e
    return by


def _episode_text(e):
    """The v1.1 instruction of one episode and why it is not usable (empty when it is)."""
    tx = e.get("text") if isinstance(e.get("text"), dict) else {}
    ins, tasks, pt = (tx.get("instruction") or "").strip(), e.get("tasks") or [], e.get("piece_tasks") or []
    why = (
        "no text.instruction"
        if not ins
        else "tasks[0] != text.instruction"
        if not tasks or tasks[0] != ins
        else "tasks[0] == piece_tasks[0] (pre-text label)"
        if pt and tasks[0] == pt[0]
        else ""
    )
    return tx, ins, why


def _args():
    """Command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--roots", required=True, help="orig=<root>,aug=<root> (the converter's --roots)")
    ap.add_argument("--combo", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lang-source", default="episode_v11")
    ap.add_argument("--table", default="", help="optional oracle: the v1.1 text table (texts.jsonl)")
    ap.add_argument("--pair", default="", help="kinematic pair: task_text from OUR text-filled manifest (same scene)")
    a = ap.parse_args()
    return a


def _read_manifest(a):
    """Read the input manifest (required columns) and refuse an out dir that already holds an output."""
    roots = dict(r.split("=", 1) for r in a.roots.split(","))
    outs = ["manifest_all.tsv", "text_report.json", "texts.sha256"]
    os.makedirs(a.out_dir, exist_ok=True)
    with open(a.manifest) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cols, rows = list(rd.fieldnames or []), list(rd)
    for c in ("dataset", "root", "clip", "round", "look", "variant", "base_clip", "axis", "task_text", "lang_source"):
        if c not in cols:
            die(f"manifest column {c} missing")
    axes = sorted({r["axis"] for r in rows})
    outs += [f"manifest_{x.lower()}.tsv" for x in axes]
    have = [o for o in outs if os.path.exists(os.path.join(a.out_dir, o))]
    if have:
        die(f"{a.out_dir} already holds {have} — use a new --out-dir")
    return roots, cols, rows, axes


def _texts_for_rows(a, rows, roots):
    """Text of every row: our row of the same scene (--pair) or the episode v1.1 instruction; misses are listed."""
    cache: dict[str, dict] = {}
    miss, out_rows, prompt = [], [], collections.Counter()
    ours = {}
    if a.pair:
        with open(a.pair) as fh:
            for x in csv.DictReader(fh, delimiter="\t"):
                k = (x["dataset"], x["root"], x["clip"], x["round"], x["look"])
                if k in ours:
                    die(f"--pair {a.pair}: scene {k} twice")
                ours[k] = x
    used = set()
    for r in rows:
        if a.pair:  # kinematic pair: our row of the same scene owns the text
            if not r["root"].startswith("kin_"):
                die(f"--pair: row root {r['root']!r} is not a kin_ root ({r['clip']})")
            k = pair_key(r)
            x = ours.get(k)
            if x is None or not x["task_text"].strip():
                miss.append(("/".join(k), "no scene of ours (--pair)"))
                continue
            used.add(k)
            prompt[f"pair:{x['lang_source']}"] += 1
            out_rows.append(dict(r, task_text=x["task_text"], lang_source=x["lang_source"] + "_pair"))
            continue
        src = os.path.join(roots[r["root"]], "data", r["dataset"], "collect", a.combo)
        if src not in cache:
            by = _episodes_by_key(src)
            cache[src] = by
        e = cache[src].get((r["clip"], r["round"], r["look"]))
        key = f"{r['dataset']}/{r['root']}/{r['clip']}/{r['round']}/{r['look']}"
        if e is None:
            miss.append((key, "episode not in episodes.jsonl"))
            continue
        tx, ins, why = _episode_text(e)
        if why:
            miss.append((key, why))
            continue
        prompt[f"{tx.get('model')}:{tx.get('prompt_sha')}"] += 1
        out_rows.append(dict(r, task_text=ins, lang_source=a.lang_source))
    return miss, out_rows, prompt, ours, used


def _report(a, rows, out_rows, miss, prompt, ours, used):
    """The text report: inputs and hashes, coverage, misses, the text rule and the prompt counts."""
    n = len(rows)
    rep = {
        "manifest": os.path.abspath(a.manifest),
        "manifest_sha256": sha256_file(a.manifest),
        "rows": n,
        "covered": len(out_rows),
        "misses": [list(m) for m in miss],
        "lang_source": a.lang_source,
        "rule": (
            "task_text = our row of the same scene (--pair · our key = pair_root · pair_round when the row "
            "has them, else root without kin_)"
            if a.pair
            else "task_text = final-root episode tasks[0] == text.instruction != piece_tasks[0] · "
            "key = converter's (clip, round, look_name)"
        ),
        "prompt_model_sha": dict(prompt),
        "pair": {
            "ours": os.path.abspath(a.pair),
            "ours_rows": len(ours),
            "paired": len(used),
            "ours_unpaired": len(ours) - len(used),
        }
        if a.pair
        else None,
    }
    return n, rep


def _position_words(out_rows, axes, rep):
    """Position words in the texts, moved placements vs original, raw and without hand phrases."""
    pw = {}
    for kind, sel in (("moved", lambda x: bool(x["variant"].strip())), ("orig", lambda x: not x["variant"].strip())):
        sub = [x for x in out_rows if sel(x)]
        raw = [x for x in sub if pos_words(x["task_text"], False)]
        noh = [x for x in sub if pos_words(x["task_text"], True)]
        wc = collections.Counter(w for x in sub for w in set(pos_words(x["task_text"], True)))
        pw[kind] = {
            "rows": len(sub),
            "with_pos_raw": len(raw),
            "with_pos_no_hand": len(noh),
            "by_axis": {
                ax: [sum(1 for x in sub if x["axis"] == ax), sum(1 for x in noh if x["axis"] == ax)] for ax in axes
            },
            "word_counts_no_hand": dict(wc.most_common()),
        }
    rep["position_words"] = dict(words=list(POS_WORDS), hand_phrase_rule=HAND_RE.pattern, **pw)
    return pw


def _table_oracle(a, out_rows, rep):
    """--table: compare every text with the v1.1 text table (equal, differ, absent)."""
    if a.table:
        tab = collections.defaultdict(set)
        with open(a.table) as fh:
            for ln in fh:
                if ln.strip():
                    t = json.loads(ln)
                    tab[(t.get("dataset"), t.get("clip"))].add((t.get("instruction") or "").strip())
        eq = df = ab = 0
        diffs = []
        for x in out_rows:
            s = tab.get((x["dataset"], x["base_clip"]))
            if not s:
                ab += 1
            elif x["task_text"] in s:
                eq += 1
            else:
                df += 1
                diffs.append([x["dataset"], x["clip"], x["task_text"][:120], sorted(s)[0][:120]])
        rep["table_oracle"] = {
            "table": os.path.abspath(a.table),
            "table_sha256": sha256_file(a.table),
            "equal": eq,
            "differ": df,
            "absent": ab,
            "differ_rows": diffs[:50],
        }
        print(
            f"{TAG} table oracle {os.path.basename(a.table)} sha={rep['table_oracle']['table_sha256'][:12]} "
            f"equal={eq} differ={df} absent={ab}",
            flush=True,
        )


def _write_report(a, rep, miss, out_rows, n, prompt, pw):
    """Write the report, print the coverage and position-word lines; any miss dies before a manifest is written."""
    with open(os.path.join(a.out_dir, "text_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    for m in miss[:20]:
        print(f"{TAG} MISS {m[0]} · {m[1]}", flush=True)
    print(
        f"{TAG} coverage {len(out_rows)}/{n} ({100.0 * len(out_rows) / max(n, 1):.1f} %) · misses {len(miss)} · "
        f"model:prompt {dict(prompt)}",
        flush=True,
    )
    for kind in ("moved", "orig"):
        p = pw[kind]
        print(
            f"{TAG} position words {kind}: rows {p['rows']} · with position word raw {p['with_pos_raw']} · "
            f"without hand phrases {p['with_pos_no_hand']} · by axis [rows, no-hand hits] {p['by_axis']}",
            flush=True,
        )
    if miss:
        die(
            f"{len(miss)}/{n} rows without a v1.1 episode text — report {a.out_dir}/text_report.json "
            "(no manifest written)"
        )


def _write_manifests(a, cols, out_rows, axes, n):
    """Write manifest_all.tsv and one manifest per axis (temp + rename) and the sha256 list."""

    def write(name: str, rs: list[dict]) -> None:
        p = os.path.join(a.out_dir, name)
        with open(p + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rs)
        os.replace(p + ".tmp", p)

    write("manifest_all.tsv", out_rows)
    for ax in axes:
        write(f"manifest_{ax.lower()}.tsv", [x for x in out_rows if x["axis"] == ax])
    with open(os.path.join(a.out_dir, "texts.sha256"), "w") as fh:
        for o in ["manifest_all.tsv"] + [f"manifest_{x.lower()}.tsv" for x in axes] + ["text_report.json"]:
            fh.write(f"{sha256_file(os.path.join(a.out_dir, o))}  {o}\n")
    print(
        f"{TAG} DONE rows {n} · covered {len(out_rows)} · axes "
        f"{ {ax: sum(1 for x in out_rows if x['axis'] == ax) for ax in axes} } · distinct texts "
        f"{len({x['task_text'] for x in out_rows})} · out {a.out_dir}",
        flush=True,
    )


def main() -> int:
    a = _args()
    roots, cols, rows, axes = _read_manifest(a)
    miss, out_rows, prompt, ours, used = _texts_for_rows(a, rows, roots)
    n, rep = _report(a, rows, out_rows, miss, prompt, ours, used)
    pw = _position_words(out_rows, axes, rep)
    _table_oracle(a, out_rows, rep)
    _write_report(a, rep, miss, out_rows, n, prompt, pw)
    _write_manifests(a, cols, out_rows, axes, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
