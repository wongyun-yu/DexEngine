#!/usr/bin/env python3
"""E5 kinematic data — the **only owner** of the cell (task × dataset × hand use) samples.

Rule: per cell only clips whose source placement passed reach are candidates (--gate reach); placement pipeline takes
  every pool clip (--gate none). Candidate order = sha256(f"{SEED}\\0{axis}\\0{ds}\\0{hand_use}\\0{clip}") ascending
  (seed 42, deterministic, independent of processing order). Clips are added from the front until the target episodes
  (the cell count of our training manifest) are filled. One clip's episodes = LOOKS (5) × (1 source + remaining
  variants). The variant count comes from the variant branch (prep.py --variants); a clip not yet known counts
  augment policy.PER_CLIP (optimistic; recounted when variant results arrive). When candidates run out, what exists is
  used and the shortfall is written (manifest).
Output (--out folder): sample.tsv (cell, clip, order, variant result present, remaining variants, episodes) · cells.tsv
  (per cell target, count, shortfall, candidates, clips used) · next_variants.tsv (sampled clips without a variant
  result yet — the next variant branch's input: ds emb clip). Marker [KIN_SAMPLE] DONE …
Split mode (--split-orig-var, default off): one look D; per cell the source episodes orig_eps and variant episodes
  var_eps are targeted separately (our source/variant ratio). One clip = 1 source D + up to its variant D count
  (variant result n_kept, else PER_CLIP) — from the front, a clip contributes its source while sources are short and
  its variants while variants are short. sample.tsv gains use_orig (0/1) and use_variants (variants to use, in draw
  order); cells.tsv gains source and variant targets, counts and shortfalls. Marker [KIN_SAMPLE] split mode (look D,
  source/variant apart).
  python3 experiments/policies/kin_data/sample.py --clips <kin_cells_clips.tsv> --targets <counts.tsv>
      --variants-glob '<jsonl>' --out <folder>
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from pathlib import Path

MARK = "[KIN_SAMPLE]"
SEED = 42
LOOKS = 5


def order_key(axis: str, ds: str, hu: str, clip: str) -> str:
    return hashlib.sha256(f"{SEED}\0{axis}\0{ds}\0{hu}\0{clip}".encode()).hexdigest()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--clips", required=True, help="experiments/policies/kin_data/cells.py --clips-tsv output (cell · reach pass)"
    )
    ap.add_argument("--targets", required=True, help="our training manifest counts.tsv")
    ap.add_argument(
        "--variants-glob", action="append", default=[], help="variant-branch result jsonl glob (repeatable)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--hold",
        action="append",
        default=[],
        help="hold tsv (kin_layer --hold-out output hold.tsv: ds emb tree reason) — those clips leave the candidates"
        " (cell table held)",
    )
    ap.add_argument(
        "--gate",
        required=True,
        choices=("reach", "none"),
        help="reach = only clips passing reach at the source placement (placement clipside) · none = every pool clip ("
        "placement pipeline) — no default",
    )
    ap.add_argument(
        "--split-orig-var",
        action="store_true",
        help="split mode (one look D · fills the targets' orig_eps and var_eps separately) — default off = unchanged",
    )
    return ap.parse_args()


def _variant_results(a) -> dict:
    """Successful variant-branch results keyed by (dataset, clip), from every jsonl matching --variants-glob."""
    var = {}
    for f in sorted({f for g in a.variants_glob for f in glob.glob(g)}):
        with open(f) as fh:
            for ln in fh:
                if ln.strip():
                    d = json.loads(ln)
                    if d.get("ok") and d.get("mode") == "variants":
                        var[(d["ds"], d["clip"])] = d
    return var


def _held_reasons(a) -> dict[tuple, str]:
    """Held clips (--hold tables) keyed by (dataset, clip) → reason."""
    held: dict[tuple, str] = {}
    for f in a.hold:
        with open(f) as fh:
            for h in csv.DictReader(fh, delimiter="\t"):
                held[(h["ds"], h["tree"].removesuffix("_pl_rt"))] = h["reason"]
    return held


def _check_gate(a, rows: list[dict]) -> None:
    """--gate must match the clip table's placement (placement_src present = pipeline = gate none)."""
    srcs = {bool(r.get("placement_src")) for r in rows}
    if srcs != {a.gate == "none"}:
        raise SystemExit(
            f"{MARK} --gate {a.gate} does not match the clip table placement (placement_src present = pipeline = none)"
        )


def _candidates(a, rows: list[dict], held: dict[tuple, str]) -> tuple[dict[tuple, list], dict[tuple, int]]:
    """Candidate clips per cell (gate passed, not held) and the held count per cell."""
    cand: dict[tuple, list] = {}
    nheld: dict[tuple, int] = {}
    for r in rows:
        if a.gate == "none" or r["reach_pass"] == "1":
            k = (r["axis"], r["dataset"], r["hand_use"])
            if (r["dataset"], r["clip"]) in held:
                nheld[k] = nheld.get(k, 0) + 1
                continue
            cand.setdefault(k, []).append(r)
    return cand, nheld


def _fill_cells(tg: dict, cand: dict, nheld: dict, var: dict, per: int) -> tuple[list, list, list]:
    """Add clips per cell in candidate order until the target episodes are filled; returns sample, cells, next rows."""
    sample, cells, nxt = [], [], []
    for key in sorted(set(tg) | set(cand)):
        t = tg.get(key, 0)
        cs = sorted(cand.get(key, []), key=lambda r: order_key(*key, r["clip"]))
        got, used = 0, 0
        for i, r in enumerate(cs):
            if got >= t:
                break
            v = var.get((r["dataset"], r["clip"]))
            nv = int(v["n_kept"]) if v else per
            eps = LOOKS * (1 + nv)
            sample.append(
                {
                    "axis": key[0],
                    "dataset": key[1],
                    "hand_use": key[2],
                    "clip": r["clip"],
                    "emb": r["emb"],
                    "order": i,
                    "variants_known": int(bool(v)),
                    "n_variants": nv,
                    "episodes": eps,
                    "kept": " ".join(v["kept"]) if v else "",
                }
            )
            if not v:
                nxt.append({"ds": r["dataset"], "emb": r["emb"], "clip": r["clip"], "axis": key[0]})
            got += eps
            used += 1
        cells.append(
            {
                "axis": key[0],
                "dataset": key[1],
                "hand_use": key[2],
                "target_eps": t,
                "counted_eps": got,
                "short_eps": max(0, t - got),
                "candidates": len(cs),
                "held": nheld.get(key, 0),
                "used_clips": used,
                "unknown_variants": sum(1 for s in sample[-used:] if not s["variants_known"]) if used else 0,
            }
        )
    return sample, cells, nxt


def _write_outputs(a, sample: list[dict], cells: list[dict], nxt: list[dict]) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("sample.tsv", sample), ("cells.tsv", cells), ("next_variants.tsv", nxt)):
        with open(a.out / name, "w", newline="") as fh:
            if data:
                w = csv.DictWriter(fh, list(data[0]), delimiter="\t", lineterminator="\n")
                w.writeheader()
                w.writerows(data)
            else:
                fh.write("ds\temb\tclip\taxis\n" if name == "next_variants.tsv" else "")


def main() -> int:
    a = _parse_args()
    if a.split_orig_var:
        return split_main(a)
    from dexengine.record.augment import policy as AP

    per = int(AP.PER_CLIP)
    with open(a.clips) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(a.targets) as fh:
        tg = {(r["axis"], r["dataset"], r["hand_use"]): int(r["episodes"]) for r in csv.DictReader(fh, delimiter="\t")}
    var = _variant_results(a)
    held = _held_reasons(a)
    _check_gate(a, rows)
    cand, nheld = _candidates(a, rows, held)
    sample, cells, nxt = _fill_cells(tg, cand, nheld, var, per)
    _write_outputs(a, sample, cells, nxt)
    tot = sum(c["counted_eps"] for c in cells)
    short = sum(c["short_eps"] for c in cells)
    print(
        f"{MARK} DONE cells {len(cells)} · sampled clips {len(sample)} · counted episodes {tot} · short {short} · "
        f"samples without a variant result {len(nxt)} · seed {SEED} → {a.out}",
        flush=True,
    )
    return 0


def _held_set(a) -> set:
    """Held clips (--hold tables) as (dataset, clip) keys."""
    held = set()
    for f in a.hold:
        with open(f) as fh:
            held |= {(h["ds"], h["tree"].removesuffix("_pl_rt")) for h in csv.DictReader(fh, delimiter="\t")}
    return held


def _fill_split_cells(tg: dict, cand: dict, var: dict, per: int) -> tuple[list, list, list]:
    """Split mode: per cell, sources fill orig_eps and variants fill var_eps, clips taken in candidate order."""
    sample, cells, nxt = [], [], []
    for key in sorted(tg):
        to, tv = tg[key]
        cs = sorted(cand.get(key, []), key=lambda r: order_key(*key, r["clip"]))
        go = gv = used = 0
        for i, r in enumerate(cs):
            if go >= to and gv >= tv:
                break
            v = var.get((r["dataset"], r["clip"]))
            nv = int(v["n_kept"]) if v else per
            uo = 1 if go < to else 0
            uv = min(nv, tv - gv)
            if uo == 0 and uv == 0:
                continue
            sample.append(
                {
                    "axis": key[0],
                    "dataset": key[1],
                    "hand_use": key[2],
                    "clip": r["clip"],
                    "emb": r["emb"],
                    "order": i,
                    "placement_src": r["placement_src"],
                    "variants_known": int(bool(v)),
                    "n_variants": nv,
                    "use_orig": uo,
                    "use_variants": uv,
                    "episodes": uo + uv,
                    "kept": " ".join(v["kept"]) if v else "",
                }
            )
            if not v and uv:
                nxt.append({"ds": r["dataset"], "emb": r["emb"], "clip": r["clip"], "axis": key[0]})
            go, gv, used = go + uo, gv + uv, used + 1
        cells.append(
            {
                "axis": key[0],
                "dataset": key[1],
                "hand_use": key[2],
                "target_orig": to,
                "target_var": tv,
                "got_orig": go,
                "got_var": gv,
                "short_orig": to - go,
                "short_var": tv - gv,
                "candidates": len(cs),
                "used_clips": used,
            }
        )
    return sample, cells, nxt


def split_main(a) -> int:
    """Split mode — the module docstring's split rule. Candidates, order, hold and placement gate follow the default
    mode (order_key, --hold, --gate).
    """
    from dexengine.record.augment import policy as AP

    per = int(AP.PER_CLIP)
    with open(a.clips) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    with open(a.targets) as fh:
        tr = list(csv.DictReader(fh, delimiter="\t"))
    if not tr or not {"orig_eps", "var_eps"} <= set(tr[0]):
        raise SystemExit(f"{MARK} --split-orig-var needs orig_eps and var_eps columns in targets")
    tg = {(r["axis"], r["dataset"], r["hand_use"]): (int(r["orig_eps"]), int(r["var_eps"])) for r in tr}
    var = _variant_results(a)
    held = _held_set(a)
    _check_gate(a, rows)
    cand: dict[tuple, list] = {}
    for r in rows:
        if (a.gate == "none" or r["reach_pass"] == "1") and (r["dataset"], r["clip"]) not in held:
            cand.setdefault((r["axis"], r["dataset"], r["hand_use"]), []).append(r)
    print(
        f"{MARK} split mode (look D · source/variant apart) · seed {SEED} · unknown variant count → {per}", flush=True
    )
    sample, cells, nxt = _fill_split_cells(tg, cand, var, per)
    _write_outputs(a, sample, cells, nxt)
    print(
        f"{MARK} DONE split mode cells {len(cells)} · sampled clips {len(sample)} · source "
        f"{sum(c['got_orig'] for c in cells)}/"
        f"{sum(c['target_orig'] for c in cells)} · variants "
        f"{sum(c['got_var'] for c in cells)}/{sum(c['target_var'] for c in cells)} · "
        f"samples without a variant result {len(nxt)} → {a.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
