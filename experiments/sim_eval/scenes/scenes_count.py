#!/usr/bin/env python3
"""E5 scene-set check (set differences, not counts) and scene list writer (standard library; reads, writes only inside
the prepared folders).

  test  expected = every (dataset, clip) with split=test in e5_split.tsv → key = clip tree name · built =
        <test stage>/clips/*/job.json
  aux   expected = every aux_holdout.tsv row (aug_clip, look) → key = <aug_clip>.<look> · built = <aux
  stage>/clips/*/job.json PASS needs both differences (missing, extra) empty, and each job.json's (dataset, clip, task,
  hand use) equal to the table.
Output (each stage folder): scenes.tsv (key, dataset, task, hand use, start rule, scene tree, variant k, dx, dy, yaw,
look) and scenes_manifest.json (input sha256 — split, auxiliary table, episodes JSONL, prep commit, check counts).
Marker [E5_SCENES] PASS|FAIL …"""

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha(p) -> str:
    """sha256 of a file."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jobs_of(stage: Path) -> dict:
    """Scene key → job.json of a stage folder."""
    return {
        d.name: json.loads((d / "job.json").read_text())
        for d in sorted((stage / "clips").iterdir())
        if (d / "job.json").is_file()
    }


def write(stage: Path, got: dict, man: dict) -> None:
    """Write scenes.tsv and scenes_manifest.json into the stage folder."""
    cols = ["key", "dataset", "axis", "hand_use", "start_rule", "scene_clip", "k", "dx_m", "dy_m", "yaw_deg", "look"]
    with open(stage / "scenes.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for k, j in sorted(got.items()):
            v = j.get("variant") or {}
            p = v.get("placement") or {}
            xy = p.get("xy_shift_m") or ["", ""]
            w.writerow(
                [
                    k,
                    j["dataset"],
                    j["axis"],
                    j["hand_use"],
                    j["episode"].get("start_rule"),
                    j["clip"],
                    v.get("k", ""),
                    xy[0],
                    xy[1],
                    p.get("yaw_deg", ""),
                    v.get("look", "D"),
                ]
            )
    (stage / "scenes_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))


def _parse_args():
    """Command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--test-stage", type=Path, required=True)
    ap.add_argument("--aux-stage", type=Path, required=True)
    ap.add_argument("--test-episodes", type=Path, required=True)
    ap.add_argument("--aux-episodes", type=Path, required=True)
    a = ap.parse_args()
    return a


def _read_sets(a):
    """Wanted and staged job sets of the test split and the aux set."""
    with open(a.split) as fh:
        sp = list(csv.DictReader(fh, delimiter="\t"))
    want_t = {f"{r['clip']}_pl_rt": r for r in sp if r["split"] == "test"}
    got_t = jobs_of(a.test_stage)
    with open(a.aux) as fh:
        aux = list(csv.DictReader(fh, delimiter="\t"))
    want_a = {f"{r['aug_clip']}.{r['look']}": r for r in aux}
    got_a = jobs_of(a.aux_stage)
    return want_t, got_t, aux, want_a, got_a


def _compare_sets(want_t, got_t, want_a, got_a, ok):
    """Compare wanted and staged jobs per set; print and return the per-set result and the overall ok."""
    res = {}
    for name, want, got in (("test", want_t, got_t), ("aux", want_a, got_a)):
        miss, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        bad = []
        for k in set(want) & set(got):
            w, j = want[k], got[k]
            if (j["dataset"], j["axis"], j["hand_use"]) != (w["dataset"], w["axis"], w["hand_use"]):
                bad.append(k)
            if name == "test" and j["base"] != w["clip"]:
                bad.append(k)
            if name == "aux" and (
                j["clip"] != w["aug_clip"]
                or (j.get("variant") or {}).get("look") != w["look"]
                or j["episode"]["split_row"]["clip"] != w["clip"]
            ):
                bad.append(k)
        cnt = collections.Counter((j["axis"], j["episode"].get("start_rule")) for k, j in got.items() if k in want)
        res[name] = {
            "want": len(want),
            "got": len(got),
            "missing": miss,
            "extra": extra,
            "mismatch": sorted(set(bad)),
            "by_axis_rule": {f"{a_}/{r_}": n for (a_, r_), n in sorted(cnt.items())},
            "by_dataset": dict(collections.Counter(j["dataset"] for k, j in got.items() if k in want)),
        }
        good = not miss and not extra and not bad
        ok &= good
        print(
            f"[E5_SCENES] {name} {'PASS' if good else 'FAIL'} expected {len(want)} · built {len(got)} · "
            f"missing {len(miss)} "
            f"{miss[:5]} · extra {len(extra)} {extra[:3]} · table mismatch {len(set(bad))}",
            flush=True,
        )
    return res, ok


def _write_tables(a, got_t, res, got_a, ok):
    """Write the two count tables and print the summary; rc 1 on any mismatch."""
    commit = {str(s): json.loads((s / "prep.json").read_text()).get("tree_commit") for s in (a.test_stage, a.aux_stage)}
    base = {
        "split": str(a.split),
        "split_sha256": sha(a.split),
        "aux": str(a.aux),
        "aux_sha256": sha(a.aux),
        "prep_commit": commit,
    }
    write(
        a.test_stage,
        got_t,
        dict(base, episodes=str(a.test_episodes), episodes_sha256=sha(a.test_episodes), check=res["test"]),
    )
    write(
        a.aux_stage,
        got_a,
        dict(base, episodes=str(a.aux_episodes), episodes_sha256=sha(a.aux_episodes), check=res["aux"]),
    )
    print(
        f"[E5_SCENES] {'PASS' if ok else 'FAIL'} · split {base['split_sha256'][:12]} · aux {base['aux_sha256'][:12]}",
        flush=True,
    )
    return 0 if ok else 1


def main() -> int:
    """CLI entry; exit 1 on any set or table mismatch."""
    a = _parse_args()
    ok = True
    want_t, got_t, aux, want_a, got_a = _read_sets(a)
    res, ok = _compare_sets(want_t, got_t, want_a, got_a, ok)
    if len({r["clip"] for r in aux}) != 200 or len(aux) != 1000:
        ok = False
        print(
            f"[E5_SCENES] FAIL auxiliary table shape — clips {len({r['clip'] for r in aux})} · rows "
            f"{len(aux)} (200 × 5 expected)"
        )
    return _write_tables(a, got_t, res, got_a, ok)


if __name__ == "__main__":
    sys.exit(main())
