#!/usr/bin/env python3
"""Text -> stored datasets: does every clip of a final dataset (hand x dataset, data layer collect) have a row in
the final text table (coverage), and retext only the datasets that are fully covered (under the final-dataset
lock).

Rule: a dataset with any clip outside the table (and outside the hold list) is not loaded — the dataset writer dies
on a clip outside the table, there is no silent fallback. Each pipeline pass prints one line with the covered final
datasets and the per-dataset remaining waits (targets - table). Standalone:
  python -m dexengine.record.text.store --track <track> [--apply]
— split_final (data layer collect_text) -> one coverage line [TEXT_STORE] per dataset -> with --apply retext of the
fully covered ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.text import text_table as TT
from dexengine.record.text.api import MARK, read_jsonl


def need_of(d: Path) -> tuple[set, int]:
    """(clip keys, episode count) of a final dataset; key rule = order.ep_clip (placement variant -> source clip)."""
    from dexengine.record.text import order as ORD

    eps = [json.loads(x) for x in (Path(d) / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
    return {ORD.ep_clip(e) for e in eps}, len(eps)


#: Hold list next to the table (header + rows with the clip in column 2): those clips are stored with their source
#: label only. Optional, written by hand.
HOLD_FILE = "merge_rejects_reasons.tsv"


def hold_of(final: Path) -> set:
    """Held clip keys: the clip column of HOLD_FILE next to the table (empty set without the file). Clips in the
    table are not held."""
    f = Path(final).parent / HOLD_FILE
    rows = f.read_text().splitlines()[1:] if f.is_file() else []
    return {TT.clip_key(x.split("\t")[1]) for x in rows if x.strip()}


def coverage(final: Path, datasets: list[Path]) -> dict[str, int]:
    """{final dataset folder: clips without a table row (and not held)} — a clip counted here goes to the next round
    (targets include the final datasets' clips); once its text is in the table the retext key changes and it is
    loaded."""
    have = {TT.clip_key(r["clip"]) for r in read_jsonl(final)} | hold_of(final)
    return {str(d): len(need_of(d)[0] - have) for d in datasets}


def key_of(final: Path, d: Path) -> str | None:
    """retext key of a dataset = sha of its clips' table rows (other datasets' rows do not matter), sha of its clip
    set, its episode count, sha of its held clips; a change in any of them reloads it. None when a clip is outside
    the table and the hold list (not loaded)."""
    import hashlib

    need, n = need_of(d)
    rows = {TT.clip_key(r["clip"]): r for r in read_jsonl(final)}
    out = need - set(rows)
    if out - hold_of(final):
        return None
    h = lambda b: hashlib.sha256(b).hexdigest()[:16]  # noqa: E731
    mine = "\n".join(json.dumps(rows[c], sort_keys=True, ensure_ascii=False) for c in sorted(need & set(rows)))
    return f"{h(mine.encode())}:{h(chr(10).join(sorted(need)).encode())}:{n}:{h(' '.join(sorted(out)).encode())}"


def waits(pairs: list[tuple[str, str]], final: Path) -> dict[str, int]:
    """Remaining waits per dataset = targets - table (waiting for render, unresolved and in flight together)."""
    have = {(r["dataset"], r["clip"]) for r in read_jsonl(final)}
    out: dict[str, int] = {}
    for ds, clip in pairs:
        out.setdefault(ds, 0)
        out[ds] += (ds, clip) not in have
    return out


def tick_line(root: Path, cfg: dict, final: Path | None, pairs: list[tuple[str, str]]) -> None:
    """Pass line: covered final datasets (every clip in the table) and the remaining waits per dataset."""
    if final is None:
        return
    from dexengine.record.text import finals as FIN

    cov = coverage(final, FIN.orig_finals()) if cfg["frames"] else {}
    on = sorted("/".join(Path(d).parts[-3::2]) for d, m in cov.items() if m == 0)
    gap = {"/".join(Path(d).parts[-3::2]): m for d, m in sorted(cov.items()) if m}
    w = waits(pairs, final)
    print(
        f"[TEXT_HOLD] {len(hold_of(final))} (hold list {HOLD_FILE} — those clips are stored without text)", flush=True
    )
    print(
        f"{MARK} track store on {len(on)}/{len(cov)} {on} · finals minus texts (hold excluded) "
        f"{sum(gap.values())} {gap} · remaining waits {sum(w.values())} "
        f"{ {k: v for k, v in sorted(w.items()) if v} }",
        flush=True,
    )


def retext(root: Path, cfg: dict, st: dict, final: Path | None, save) -> None:
    """Reload the final datasets (data layer collect, one per hand x dataset) whose clips are all in the final table
    (key = key_of: a dataset that grew by merged pieces reloads even with the same table)."""
    if not cfg["retext"] or final is None:
        return
    from dexengine.record.text import finals as FIN

    apply(final, FIN.final_datasets(), st["retexted"])
    save(root, st)


def apply(final: Path, datasets: list[Path], done: dict) -> list[str]:
    """retext the datasets whose key (key_of) is set and differs from the last load (under the final-dataset lock);
    updates done."""
    from dexengine.record import piece as PC

    out = []
    for d in datasets:
        key = key_of(final, d)
        if key is not None and done.get(str(d)) != key:
            a = types.SimpleNamespace(
                dataset=[str(d)], texts=str(final), off=False, lock=PC.final_lock(d)
            )  # the merge's lock, held only while swapping files
            TT.cmd_retext(a)
            if not any(r.get("retry") for r in getattr(a, "results", [])):  # a merge changed meta meanwhile: retry
                done[str(d)] = key
                out.append(str(d))
    return out


def main(argv=None) -> int:
    """CLI: split the final table, print coverage, and with --apply retext the fully covered final datasets."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True)
    ap.add_argument("--apply", action="store_true", help="retext the fully covered final datasets (else report only)")
    a = ap.parse_args(argv)
    from dexengine.record.text import finals as FIN
    from dexengine.record.text import pipeline as CTT
    from dexengine.record.text import track as TRK

    final = TRK.track_root(a.track) / "final" / "texts.jsonl"
    if not final.is_file():
        raise InputMissingError(f"{MARK} [TEXT_STORE] no table {final}")
    n = CTT.split_final(final)
    cov = coverage(final, FIN.final_datasets())
    for d, miss in cov.items():
        print(f"{MARK} [TEXT_STORE] {d} clips outside the table {miss}", flush=True)
    kf = final.parent / "retexted.json"  # load keys of this machine (same key = no reload; rerunnable)
    done = json.loads(kf.read_text()) if kf.is_file() else {}
    did = apply(final, FIN.final_datasets(), done) if a.apply else []
    if a.apply:
        kf.write_text(json.dumps(done, indent=1))
    on = sum(m == 0 for m in cov.values())
    print(
        f"{MARK} [TEXT_STORE] done table rows {sum(n.values())} · covered datasets {on}/{len(cov)} · "
        f"{f'retext {len(did)}' if a.apply else 'report only'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
