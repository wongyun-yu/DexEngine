"""A campaign bundle cell's file names, its campaign's frozen manifest copy, and the per-clip record a launch freezes.

A leaf module (standard library only): the trainer's checkpoint check (`dexengine.rl.checkpoints`) reads the cell file
names here, so a training or judge process never imports the campaign runner modules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: A bundle cell's own files; the runner, bundle_state, launch and finalize read these names.
CELL_FILES = {
    "launch": "launch.json",
    "launch_cmd": "launch_cmd.txt",
    "train_pid": "train.pid",
    "train_rc": "train.rc",
    "train_out": "train.out",
    "train_gpu": "train.gpu",
    "judge_pid": "judge.pid",
    "judge_out": "judge.out",
    "judge_starts": "judge.starts",
    "overflow": "overflow.txt",
    "finalize_tsv": "finalize.tsv",
    "finalize_log": "finalize.log",
    "finalized": "FINALIZED.json",
    "finalize_lock": ".finalize.lock",
    "render_pid": "render.pid",
    "render_out": "render.out",
    "replay_pid": "replay.pid",
    "replay_out": "replay.out",
    "attempts": "attempts",
    "clips": "clips.tsv",
    "rejudged": "rejudged",
}  # rejudged/ = rejudge attempts

#: Head of the per-clip table (lists/<id>.tsv in a manifest, clips.tsv in a finalized cell).
TABLE_HEAD = "clip\tdataset\tobj_key\tfamily\tdup\torigin\n"


def cell_file(sdir: Path, key: str) -> Path:
    """A bundle cell's file by its CELL_FILES key (an unknown key raises KeyError)."""
    return sdir / CELL_FILES[key]


def manifest_dir(cdir: Path) -> Path:
    """The frozen manifest copy inside a campaign cell."""
    return cdir / "manifest"


def bundle_files(man: dict) -> list[str]:
    """A manifest's bundle files next to it: every bundle's clip list and clip table (relative paths, in order).
    Every copy of a manifest carries this set; a bundle without both raises KeyError."""
    return [b[k] for b in man["bundles"] for k in ("list", "table")]


def bundle_datasets(b: dict) -> dict[str, int]:
    """{dataset: primary clips of that dataset} of one bundle (the input of the cell rule, paths.rl_bundle_dir). A
    bundle is one ledger axis, so every clip is the axis' dataset. Counted over primaries: a dup (a clip borrowed from
    its primary bundle) counts only in its primary bundle."""
    if len(b["clips"]) != b["n"] or b["n"] <= 0 or b["n_primary"] + len(b["dups"]) != b["n"]:
        raise ValueError(
            f"bundle {b['id']}: clips {len(b['clips'])} · n {b['n']} · primary {b['n_primary']} + dups {len(b['dups'])}"
        )
    return {b["dataset"]: b["n_primary"]}


def clip_rows(b: dict) -> list[list[str]]:
    """The bundle's per-clip rows: clip · dataset · object key · family · dup (1 = borrowed from its primary bundle) ·
    origin (its primary bundle id, '-' = primary)."""
    return [
        [c, b["dataset"], b["keys"][c], b["families"][b["keys"][c]], str(int(c in b["dups"])), b["dups"].get(c, "-")]
        for c in b["clips"]
    ]


def table_text(rows: list[list[str]]) -> str:
    """The per-clip table file text."""
    return TABLE_HEAD + "".join("\t".join(r) + "\n" for r in rows)


def sequence_record(b: dict) -> dict:
    """Which sequences a run of this bundle trains, frozen into its launch record and carried to FINALIZED.json:
    dataset · clips · dups · per-clip rows · table sha256."""
    rows = clip_rows(b)
    return {
        "dataset": b["dataset"],
        "clips": list(b["clips"]),
        "dups": dict(b["dups"]),
        "clip_rows": rows,
        "clips_sha256": hashlib.sha256(table_text(rows).encode()).hexdigest(),
    }
