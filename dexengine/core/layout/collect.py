"""Storage of the record (data generation) stage: final dataset roots per round, piece and archive locations,
and the per-GPU session count rule of the generation runs."""

from __future__ import annotations

import os

from dexengine.core.layout.layers import render
from dexengine.core.layout.registry import DATASETS

#: How many collect jobs (one Kit-reusing worker each) share one GPU — the one home of this rule.
#: n = min(VRAM slots, CPU slots, RAM slots, COLLECT_SESSIONS_MAX), at least 1 (too few cores only makes it slower).
#: VRAM slots = counted by the caller from one job's peak.
#: CPU slots = cores per GPU × COLLECT_CPU_SHARE ÷ COLLECT_SESSION_CORES. Basis (one GPU · 3 workers · 30 sessions ·
#: taskset): 64 cores → 346 s · 16 → 382 · 12 → 366 · 8 → 363 · 6 → 448 · 4 → 571 — the knee is 8 (3 workers).
#: One worker = Kit 1.65 + encoder 0.13 cores. 8 cores per GPU (cluster jobs also get 8 CPUs per GPU).
#: RAM slots = host RAM per GPU ÷ COLLECT_SESSION_RAM_GB — one collect job's RSS p90 is 10.7 GB (measured) → 11.
#: Cap 3 = render throughput saturates at 48 GB (4 workers = 3 workers, measured on the same 10 pieces).
#: Machine fact (measured): cores per GPU = host cores ÷ GPU count — an 8-GPU host 64 ÷ 8 · a 2-GPU host 24 ÷ 2.
#: Readers: per-host collect slot counts (VRAM per slot) and a cluster lane's cap (each job
#: calls this function with its job CPU and VRAM).
COLLECT_SESSION_CORES, COLLECT_CPU_SHARE, COLLECT_SESSIONS_MAX, COLLECT_SESSION_RAM_GB = 2.0, 0.8, 3, 11
#: Machine fact (measured): per GPU (cores, host RAM GB) — 64 cores · 503 GB ÷ 8 GPUs · 24 cores · 125 GB ÷ 2 GPUs.
COLLECT_HOST_PER_GPU = {"8-GPU host": (64 / 8, 503 / 8), "2-GPU host": (24 / 2, 125 / 2)}


def collect_sessions_per_gpu(vram_slots: int, cores_per_gpu: float, ram_gb_per_gpu: float) -> int:
    """Collect jobs per GPU (rule in the comment above). Dies on 0 VRAM slots, RAM for not even one job, or 0 cores."""
    ram = int(ram_gb_per_gpu // COLLECT_SESSION_RAM_GB)
    if int(vram_slots) < 1 or ram < 1 or not cores_per_gpu > 0:
        raise ValueError(
            f"collect_sessions_per_gpu: VRAM slots {vram_slots} · cores {cores_per_gpu} · "
            f"RAM {ram_gb_per_gpu} GB — 0 jobs"
        )
    cpu = int(cores_per_gpu * COLLECT_CPU_SHARE // COLLECT_SESSION_CORES)
    return max(1, min(int(vram_slots), cpu, ram, COLLECT_SESSIONS_MAX))


#: Per-GPU count over the measured hosts (the slot gate checks VRAM separately → VRAM slots = the cap) — the
#: narrower of the two hosts.
COLLECT_SESSIONS_PER_GPU = min(
    collect_sessions_per_gpu(COLLECT_SESSIONS_MAX, c, r) for c, r in COLLECT_HOST_PER_GPU.values()
)

# ── Record storage roots (LayoutConfig: configs/layout.yaml or DEXENGINE_LAYOUT_CONFIG). The release keeps everything
# inside the data root; a setup whose disks split the record data names its roots in its own layout file.
#: Rounds of the final collect datasets; both share the layout below the root — <root>/LAYERS['collect'].
COLLECT_ROUNDS = ("orig", "aug")


def collect_round_roots() -> dict[str, str | None]:
    """Final collect root per round: None = that machine's data root (dexengine.core.paths.main_root) · a string =
    that absolute root (LayoutConfig.collect_aug_root for the augmented round). The writer (the merge step) and the
    readers (scene_layout · progress · audits) get the place from collect_final only — an unknown round dies."""
    from dexengine.core.layout.roots import layout_config  # noqa: PLC0415

    return {"orig": None, "aug": layout_config().collect_aug_root or None}


def traj_archive_root() -> str:
    """Where final trajectories are archived: a re-render that deletes old videos / depth keeps the trajectories as
    evidence (stage 1 keys under stage1/, each piece's small records under jobs/). LayoutConfig.collect_traj_archive,
    else <data root>/archive/collect_traj."""
    from dexengine.core.layout.roots import layout_config  # noqa: PLC0415

    if layout_config().collect_traj_archive:
        return layout_config().collect_traj_archive
    from dexengine.core.locations.roots import main_root  # noqa: PLC0415 — locations imports this package

    return os.path.join(str(main_root()), "archive", "collect_traj")


#: Under it, stage 1 trajectories (the collect cache's <out>/stage1/<key>/) are archived as hand → dataset → key →
#: {COLLECT_STAGE1_FILES}. The key is the cache key (clip · checkpoint · rollout count · code fingerprint). A piece
#: that re-ran the same key on another machine can hold different content (rollouts) — only then does that piece's
#: whole key go to the variant place collect_stage1_variant(…, piece_id) (the old place stays byte-identical; equal
#: content stays in the old place). Writers = the salvage of running jobs and the collect lane that receives a piece's
#: stage1 — the place rules are the two functions below.
COLLECT_STAGE1_FILES = ("rollouts.npz", "record.json")


def collect_stage1_archive(hand: str, ds: str, key: str) -> str:
    """Stage 1 archive place <traj_archive_root()>/stage1/<hand>/<ds>/<key> — dies on an unknown dataset or an empty
    field."""
    if ds not in DATASETS or not hand or not key or "/" in hand + key:
        raise KeyError(f"stage 1 archive place — hand {hand!r} · dataset {ds!r} (DATASETS) · key {key!r}")
    return os.path.join(traj_archive_root(), "stage1", hand, ds, key)


def collect_stage1_variant(hand: str, ds: str, key: str, piece_id: str) -> str:
    """Archive place of a piece that carried the same stage 1 key with different content
    <collect_stage1_archive(hand, ds, key)>/<piece_id> — the piece's whole key goes here only when at least one file
    of that key differs from the old place (writer = the merge transport · readers = collect_seeds.seed_scan ·
    collect_seed_use.build). Dies on an empty piece id or one containing '/'."""
    if not piece_id or "/" in piece_id or piece_id in (".", ".."):
        raise KeyError(f"stage 1 variant place — piece {piece_id!r}")
    return os.path.join(collect_stage1_archive(hand, ds, key), piece_id)


def collect_job_archive(hand: str, ds: str, piece_id: str) -> str:
    """Archive place of each piece's small records (the carried files outside the final and stage1 — render_meta ·
    tree_ref · preview images …) <traj_archive_root()>/jobs/<hand>/<ds>/<piece_id> — below it, the bundle out's
    relative paths as they are. Dies on an unknown dataset or an empty field."""
    if ds not in DATASETS or not hand or not piece_id or "/" in hand + piece_id:
        raise KeyError(f"piece record archive place — hand {hand!r} · dataset {ds!r} (DATASETS) · piece {piece_id!r}")
    return os.path.join(traj_archive_root(), "jobs", hand, ds, piece_id)


def collect_job_archive_root(rnd: str) -> str:
    """The new-write root of that round's piece record archive (the jobs/ part of collect_job_archive; both rounds
    share traj_archive_root()) — dies on an unknown round."""
    if rnd not in COLLECT_ROUNDS:
        raise KeyError(f"piece record archive root — round {rnd!r} (rounds {sorted(COLLECT_ROUNDS)})")
    return traj_archive_root()


def collect_job_archive_roots(rnd: str) -> tuple[str, ...]:
    """The reader's place list = (new root, *older read-only places — LayoutConfig.collect_archive_old_aug for the
    augmented round) — searched in this order."""
    from dexengine.core.layout.roots import layout_config  # noqa: PLC0415

    return (collect_job_archive_root(rnd), *(layout_config().collect_archive_old_aug if rnd == "aug" else ()))


def collect_final(ds: str, combo: str, rnd: str, data_root: str) -> str:
    """Final collect place (absolute) of round `rnd` = round root (else data_root) / LAYERS['collect'] · KeyError if
    the round is unknown."""
    roots = collect_round_roots()
    if rnd not in roots:
        raise KeyError(f"collect round {rnd!r} — only {sorted(roots)}")
    return os.path.join(roots[rnd] or data_root, render("collect", ds, combo=combo))
