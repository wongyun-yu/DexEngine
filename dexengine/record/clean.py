"""Cleanup of the heavy intermediate outputs of a record bundle: only bundles that passed the store and the check
(driver_check.verify; the driver's run_bundle calls this after verify). Unchecked or failed bundles stay as they are
(for investigation).

Removed (names stated by their makers; no copied list):
  stage2/<key>/<clip>/render.episode.media_files   old per-trajectory render (bundles of older trees): three videos and
                                                   depth (· hand-to-head marking); byte copies in the dataset
  stage2c/<key>/<clip>/<traj>/<look>/media_files   per-clip render (render.clip, per episode); copies in the dataset
  <round place>/preview/store.PREVIEW_VIDEO_DIR/*  preview videos (copies of the dataset videos, downscaled, depth
                                                   colour-map video); round place = piece.round_dir (that bundle's
                                                   round)
Kept:
  all of stage1/: the only source of the chosen trajectory (rollouts.npz, the selection table of every rollout). A
    re-render (when camera or torso rules change) needs the same trajectory, and a stage-1 rerun is not a bit-exact
    reproduction (judge band ±1.5 pp) · size ≈ 6 % of the dataset.
  the small records of stage2 (render_meta.json, session.log, dryrun.out, render_code.txt) · preview stills · json
    (for people; the kept preview is capped at PREVIEW_KEEP_MB_PER_EP per episode; over it nothing is removed and it
    dies).
Path guard: every file to remove must not be a symlink and must resolve under the bundle out; any violation removes
  nothing and dies. Session logs (KEEP_NAMES: session.log) in the plan die: the job entry checks the layer markers of
  every session after the cleanup (lane session-check).
  Per-file unlink (no tree removal) · only an empty preview video folder is rmdir'd.
Record: <round place>/cleanup.json + the return value (the driver puts it in result.json cleanup) · the removed rows
  leave preview/manifest.json (moved to the removed field) · marker [COLLECT_CLEAN].
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dexengine.core import paths as P  # clip name normalisation (clip_name)
from dexengine.record.render import clip as CCL  # owner of the per-clip render folder form
from dexengine.record.render import episode as CRN  # owner of the render output names (MEDIA)
from dexengine.record.store import store as ST  # owner of the preview video folder name

MARK = "[COLLECT_CLEAN]"
PREVIEW_KEEP_MB_PER_EP = 2.0
KEEP_NAMES = ("session.log",)  # names the cleanup never removes; the layer marker proof (lane session_logs) reads them


#: canonical statement of carrying and keeping (a separate carry list would silently lose the stage1, tree_ref and
#: off-screen records missing from it). Globs relative to the bundle out (fnmatch: '*' crosses '/'). Each job file is
#: either CARRY (carried to the final or the keep place) or to be removed (plan: render media, preview videos). Neither
#: = unclassified: production jobs carry it as CARRY and the merge raises [CARRY_UNCLASSIFIED]; smoke jobs
#: (self-tests) die. Readers = carry, merge, backup, oracle (classify).
CARRY = (
    "piece.json",
    "rounds/*/piece.json",
    "rounds/*/dataset/*",  # merged into the final
    "rounds/*/cleanup.json",
    "rounds/*/preview/frames/*.jpg",
    "rounds/*/preview/*.json",
    # stage 1 = the only source
    # files next to the sha = record.cache
    "stage1/*/rollouts.npz",
    "stage1/*/record.json",
    "stage1/*/rollouts.sha256",
    "stage1/*/clips.txt",
    "stage1/*/dryrun.out",
    "stage1/*/session.log",
    "stage1/*/session.*.log",
    "rounds/*/empty.json",
    "seq1/*/tree_ref.json",
    "seq1/*/clip.txt",
    "seq1c/*/members.txt",
    "seq1c/*/*/tree_ref.json",
    "seq1c/*/*/clip.txt",
    "stage2/*/*/render_meta.json",
    "stage2/*/*/session.log",
    "stage2/*/*/session.*.log",
    "stage2c/*/*/session.*.log",
    "stage2/*/*/dryrun.out",
    "stage2/*/*/render_code.txt",
    "stage2c/*/*/session.log",
    "stage2c/*/*/dryrun.out",
    "stage2c/*/*/render_code.txt",
    "stage2c/*/*/plan.json",
    "stage2c/*/*/render_clip.json",
    "stage2c/*/*/*/*/render_meta.json",
    "stage2c/*/*/*/*/head_view.npz",  # head view for people (<= 0.12 MB per episode)
    "stage2c/*/*/_look/*",  # look choice records (swept.npz, D0 picture; <= 0.5 MB per clip)
    # view records of originally off-screen clips (render.clip_all.write_fov_records; without them the judged cells are
    # built by the general rule)
    "rounds/*/orig_fov/*.json",
    "rounds/*/record.json",
    # augment and record-only rounds (bundle --until record): what the cache reuse reads when the same command runs
    # again without --until (augment gen, rollout). '*' crosses '/', so the same names under ext/ (phys_extend) match.
    "aug/record_items.json",
    "aug/counts.json",
    "aug/clips/*/gen/variants_*.json",
    "aug/clips/*/run/run.json",
    "aug/clips/*/run/rollouts.npz",
    "aug/clips/*/run/session.log",
    "aug/clips/*/post.json",
    "aug/clips/*/traj/*/rollouts.npz",
    # shadow root: what and how (symlinks followed to real files) is owned by the shadow packer's listing
    "aug/shadow/*",
)


def classify(out: Path, rd: Path) -> dict[str, list[str]]:
    """Every file of the bundle out -> {carry, discard, unclassified} (out-relative paths). discard = plan(out, rd,
    None)."""
    import fnmatch

    out = Path(out)
    drop = {p.relative_to(out).as_posix() for p in plan(out, rd, None)}
    res: dict[str, list[str]] = {"carry": [], "discard": [], "unclassified": []}
    for d, _n, fs in os.walk(out):
        for f in fs:
            rel = (Path(d) / f).relative_to(out).as_posix()
            k = (
                "discard"
                if rel in drop
                else ("carry" if any(fnmatch.fnmatchcase(rel, g) for g in CARRY) else "unclassified")
            )
            res[k].append(rel)
    return res


FINAL_GLOBS = ("piece.json", "rounds/*/piece.json", "rounds/*/dataset/*")  # CARRY entries that go to the final (merge)
#: CARRY entries that are the augment record-only cache (under aug/; a re-render reads up to run/rollouts.npz); a setup
#: that keeps a second copy on another disk carries only them in that step and leaves them out of its other backup.
CARRY_AUG_CACHE = tuple(g for g in CARRY if g.startswith("aug/"))


def carry_dest(rel: str, hand: str, ds: str, piece_id: str) -> str | None:  # noqa: V103 — public API for a plugin merge writer
    """Destination of one CARRY (or unclassified) file: None = final (the merge puts it into the dataset and
    pieces.jsonl) · otherwise the absolute keep path (stage1/<key>/… -> data_layout.collect_stage1_archive · the rest
    -> collect_job_archive/<bundle out relative>). The writer (a plugin merge) reads this one
    function. When the same key exists in the keep place with different content, the writer moves the whole key of
    that piece to data_layout.collect_stage1_variant(…, piece_id) (this function gives the old place)."""
    import fnmatch

    if any(fnmatch.fnmatchcase(rel, g) for g in FINAL_GLOBS):
        return None
    parts = rel.split("/")
    if parts[0] == "stage1" and len(parts) >= 3:
        return os.path.join(P.data_layout.collect_stage1_archive(hand, ds, parts[1]), *parts[2:])
    return os.path.join(P.data_layout.collect_job_archive(hand, ds, piece_id), rel)


def du(p: Path) -> int:
    return sum(os.lstat(os.path.join(d, f)).st_size for d, _n, fs in os.walk(p) for f in fs) if p.is_dir() else 0


# stage-2 render roots and render folder forms: old per trajectory
STAGE2_ROOTS = (("stage2", "*/*"), (CCL.ROOT_DIR, CCL.EPISODE_GLOB))  # per clip (render.clip: a folder per episode)


def plan(out: Path, rd: Path, clips: list[str] | None) -> list[Path]:
    """Files to remove: MEDIA of the clip folders of the stage-2 roots (STAGE2_ROOTS) · files of the preview video
    folder of the round place rd. clips = this run's clips (clips of the spec part): only those clip folders (second
    field of <key>/<clip>/…, checked with paths.clip_name), so sibling parts sharing the bundle place keep their
    renders. None = the whole bundle (redraw; the caller says so)."""
    want = None if clips is None else {P.clip_name(c) for c in clips}
    files = [
        d / m
        for r, g in STAGE2_ROOTS
        for d in sorted((out / r).glob(g))
        if d.is_dir() and not d.is_symlink() and (want is None or P.clip_name(d.relative_to(out / r).parts[1]) in want)
        for m in CRN.media_files(d)
        if (d / m).exists() or (d / m).is_symlink()
    ]
    pv = rd / "preview" / ST.PREVIEW_VIDEO_DIR
    return files + (sorted(p for p in pv.rglob("*") if p.is_file() or p.is_symlink()) if pv.is_dir() else [])


def guard(out: Path, files: list[Path]) -> None:
    root = out.resolve()
    keep = [str(p) for p in files if p.name in KEEP_NAMES]
    if keep:
        raise SystemExit(
            f"{MARK} keep names {KEEP_NAMES} in the removal plan {len(keep)}: {keep[:3]} (nothing removed)"
        )
    bad = [str(p) for p in files if p.is_symlink() or not p.resolve().is_relative_to(root) or p.resolve() == root]
    if bad:
        raise SystemExit(f"{MARK} path guard: {len(bad)} symlinks or outside the bundle: {bad[:3]} (nothing removed)")


def clean(out: Path, verified: dict, rd: Path, clips: list[str] | None) -> dict:
    """verified = the driver_check.verify result · rd = the checked round place (under out) · clips = this run's clips
    (plan; None = the whole bundle). Not ok: nothing is removed."""
    out, rd = Path(out), Path(rd)
    if not rd.resolve().is_relative_to(out.resolve()):
        raise SystemExit(f"{MARK} round place {rd} is outside the bundle place {out}: nothing removed")
    if not verified.get("ok"):
        print(f"{MARK} skipped: check not passed · intermediate outputs stay · {out}", flush=True)
        return {"done": False, "reason": "check not passed"}
    files = plan(out, rd, clips)
    guard(out, files)
    pv = rd / "preview"
    drop = {p.resolve() for p in files}
    kept_pv = (
        sum(p.stat().st_size for p in pv.rglob("*") if p.is_file() and p.resolve() not in drop) if pv.is_dir() else 0
    )
    cap = PREVIEW_KEEP_MB_PER_EP * 1e6 * max(1, int(verified.get("episodes", 0)))
    if kept_pv > cap:
        raise SystemExit(
            f"{MARK} kept previews {kept_pv / 1e6:.1f} MB > cap {cap / 1e6:.1f} MB "
            f"({PREVIEW_KEEP_MB_PER_EP} MB per episode): nothing removed"
        )
    removed = []
    for p in files:
        n = p.stat().st_size
        p.unlink()
        removed.append({"path": str(p.relative_to(out)), "bytes": n})
    vd = pv / ST.PREVIEW_VIDEO_DIR
    if vd.is_dir() and not any(vd.iterdir()):
        vd.rmdir()
    man = pv / "manifest.json"
    if man.is_file():
        m = json.loads(man.read_text())
        gone = {str(f.relative_to(pv)) for f in files if f.is_relative_to(pv)}
        m["removed"] = [f for f in m.get("files", []) if f["file"] in gone]
        m["files"] = [f for f in m.get("files", []) if f["file"] not in gone]
        man.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    pre = {k: k for k in ("stage2", CCL.ROOT_DIR, "aug")} | {"preview": str(pv.relative_to(out))}
    part = {k: sum(r["bytes"] for r in removed if r["path"].startswith(v + "/")) for k, v in pre.items()}
    rec = {
        "done": True,
        "files": len(removed),
        "bytes": sum(r["bytes"] for r in removed),
        "by_part": part,
        "kept": {
            "stage1": du(out / "stage1"),
            "stage2": du(out / "stage2"),
            "stage2c": du(out / CCL.ROOT_DIR),
            "preview": du(pv),
            "dataset": du(rd / "dataset"),
        },
        "removed": removed,
    }
    (rd / "cleanup.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(
        f"{MARK} removed {len(removed)} files · {rec['bytes'] / 1e6:.1f} MB (stage2 {part['stage2'] / 1e6:.1f} · "
        f"{CCL.ROOT_DIR} {part[CCL.ROOT_DIR] / 1e6:.1f} · "
        f"preview {part['preview'] / 1e6:.1f}) · kept stage1 {rec['kept']['stage1'] / 1e6:.1f} MB · "
        f"preview {rec['kept']['preview'] / 1e6:.1f} MB · {out}",
        flush=True,
    )
    return rec
