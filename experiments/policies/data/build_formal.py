"""Build one formal policy training set (LeRobot v2.1) — ours or its kinematic pair; the two differ only in arguments.

  python -m experiments.policies.data.build_formal --kind ours --task lift|pour|sweep|all --list <smallset_train.tsv>
      --text-manifest <manifest_<task>.tsv> --roots orig=<dir>,aug=<dir> --out <dataset dir> --work <dir>
      --ffmpeg <bin> --limits <limits.json> --git-commit <sha>
  python -m experiments.policies.data.build_formal --kind kin --task … --kin-manifest <kin conversion list>
      --text-manifest <ours manifest_<task>.tsv> --roots kin_orig=<dir>,kin_aug=<dir>[,kin_<name>=<dir>] --out … --work
      …

ours: the per-clip text manifest (manifest_text.py output) must equal the selected list row for row except task_text and
lang_source (lang_source episode_v11, text non-empty); it becomes the conversion list. kin: the text comes from the ours
row of the same scene (manifest_text.py --pair, lang_source episode_v11_pair) and every ours scene must pair 1:1.
Conversion recipe (convert.py): --combo rby1_wuji2, --action-shift 1, --action-clip-limits <URDF limits>, head stereo
cameras, 4 jobs (the worker count does not change the output). Last line `[FORMAL_BUILD] DONE …` or `… FAIL …`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MARK = "[FORMAL_BUILD]"
RECIPE = ["--combo", "rby1_wuji2", "--action-shift", "1"]


def die(msg: str):
    raise SystemExit(f"{MARK} FAIL {msg}")


def check_ours(list_tsv: Path, text_tsv: Path, axis: str) -> int:
    """The text manifest equals the selected list except task_text · lang_source (episode_v11, text non-empty)."""
    with open(list_tsv) as fh:
        a = [r for r in csv.DictReader(fh, delimiter="\t") if axis == "All" or r["axis"] == axis]
    with open(text_tsv) as fh:
        b = list(csv.DictReader(fh, delimiter="\t"))
    k = [c for c in a[0] if c not in ("task_text", "lang_source")] if a else []
    bad = [i for i, (x, y) in enumerate(zip(a, b, strict=False)) if [x[c] for c in k] != [y[c] for c in k]]
    if len(a) != len(b) or bad or not a:
        die(f"list check: rows {len(a)} vs {len(b)} · differ {bad[:5]}")
    if any(y["lang_source"] != "episode_v11" or not y["task_text"].strip() for y in b):
        die("list check: lang_source/task_text")
    print(
        f"{MARK} list check {len(b)} rows · only task_text · lang_source differ · texts "
        f"{len({y['task_text'] for y in b})}"
    )
    return len(b)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=("ours", "kin"), required=True)
    ap.add_argument("--task", choices=("lift", "pour", "sweep", "all"), required=True)
    ap.add_argument("--text-manifest", required=True, help="ours per-clip text manifest (manifest_text.py)")
    ap.add_argument("--list", default="", help="ours: the selected list (smallset_train.tsv)")
    ap.add_argument("--kin-manifest", default="", help="kin: the kinematic conversion list")
    ap.add_argument("--roots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--limits", required=True, help="URDF joint limits json (hand_limits.urdf_limits)")
    ap.add_argument("--git-commit", required=True, help="code commit recorded in the dataset meta")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout-s", type=int, default=5400)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    out, work = Path(a.out), Path(a.work)
    for x in (out, work):
        if x.exists():
            die(f"{x} exists")
    work.mkdir(parents=True)
    conv_list = work / "smallset_train.tsv"
    py = [sys.executable, "-m"]
    if a.kind == "ours":
        if not a.list or a.kin_manifest:
            die("ours takes --list and no --kin-manifest")
        check_ours(Path(a.list), Path(a.text_manifest), a.task.capitalize() if a.task != "all" else "All")
        shutil.copyfile(a.text_manifest, conv_list)
    else:
        if not a.kin_manifest or a.list:
            die("kin takes --kin-manifest and no --list")
        with open(work / "text.log", "w") as lf:
            rc = subprocess.run(
                py
                + [
                    "experiments.policies.data.manifest_text",
                    "--manifest",
                    a.kin_manifest,
                    "--roots",
                    a.roots,
                    "--combo",
                    "rby1_wuji2",
                    "--out-dir",
                    str(work / "text"),
                    "--pair",
                    a.text_manifest,
                ],
                cwd=ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
            ).returncode
        if rc != 0:
            die(f"pair text rc {rc} (see {work / 'text.log'})")
        up = json.loads((work / "text/text_report.json").read_text())["pair"]["ours_unpaired"]
        if up != 0:
            die(f"pairing is not 1:1 — {up} ours scenes have no kinematic row")
        shutil.copyfile(work / f"text/manifest_{a.task}.tsv", conv_list)
    with open(conv_list) as fh:
        n = sum(1 for _ in fh) - 1
    cmd = [
        "timeout",
        str(a.timeout_s),
        *py,
        "experiments.policies.data.convert",
        "--manifest",
        str(conv_list),
        "--roots",
        a.roots,
        *RECIPE[:2],
        "--ffmpeg",
        a.ffmpeg,
        "--git-commit",
        a.git_commit,
        *RECIPE[2:],
        "--action-clip-limits",
        a.limits,
        "--jobs",
        str(a.jobs),
        "--out",
        str(out),
    ]
    with open(work / "convert.log", "w") as lf:
        subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    log = (work / "convert.log").read_text(errors="replace")
    if f"[E5_CONVERT] DONE episodes={n} " not in log:
        die(f"conversion (see {work / 'convert.log'})")
    i = json.loads((out / "meta/info.json").read_text())
    print(
        f"{MARK} DONE {a.kind} {a.task} {out} episodes {i['total_episodes']} frames {i['total_frames']} "
        f"tasks {i['total_tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
