"""Owner of the record cache keys and code fingerprints: record.bundle decides with them whether to run stage 1
(physics record) and stage 2 (render) again.

Cache (stage-1 reuse): key = sha256(sorted clip tree names · checkpoint sha256 · rollout count · code fingerprint ·
  K_TOP); code fingerprint = CODE_FILES bytes. When <out>/stage1/<key>/{rollouts.npz, record.json} exist and
  record.json is ok, it is not run again (marker [COLLECT_CACHE] hit|miss). Stage 2 lives under the same key and is
  reused only when the render fingerprint (render_fingerprint) matches.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dexengine.core.hashing import sha256_file
from dexengine.record import chain as CANON
from dexengine.record.looks import look_d as LOOK
from dexengine.record.render import markers as PM  # render fingerprint tail (only when on)
from dexengine.record.rollout import recorder as REC

ROOT = Path(__file__).resolve().parents[2]
# entry session = render side
CODE_FILES = (
    "dexengine/record/rollout/recorder.py",
    "dexengine/record/rollout/oracle_gate.py",
    "dexengine/record/session/runner.py",
    "dexengine/record/rollout/channels.py",
    "dexengine/record/rollout/partner.py",
    "dexengine/rl/eval/traj_record.py",
    "dexengine/rl/eval/traj_meta.py",
    "dexengine/rl/tasks/mdp/hand_object_contact.py",
    "dexengine/rl/eval/traj_select.py",
    "dexengine/rl/judge/rollouts.py",
    "dexengine/rl/judge/rollouts_run.py",
    "dexengine/rl/judge/rollouts_batch.py",
    "dexengine/rl/judge/rollouts_report.py",
    "dexengine/rl/judge/rollouts_types.py",
)
# stage-2 cache fingerprint
RENDER_CODE_FILES = (
    "dexengine/record/render/episode.py",
    "dexengine/record/render/episode_frames.py",
    "dexengine/record/render/rig.py",
    *LOOK.CODE_FILES,
    "dexengine/record/looks/rev.py",
    "dexengine/record/render/framing.py",
    "dexengine/record/scene/supports.py",
    "dexengine/record/scene/extras.py",
    "dexengine/record/scene/extras_apply.py",
    "dexengine/record/render/visibility.py",
    "dexengine/record/scene/extras_rules.py",
    "dexengine/record/scene/overlap.py",
    "dexengine/record/render/body.py",
    "dexengine/record/render/body_geom.py",
    "dexengine/record/render/desk_stretch.py",
    # the arm skeleton (vendor URDF · posture) the render body chain reads — spec files and their loader
    "dexengine/core/embodiment.py",
    *sorted(f"configs/robots/arms/{p.name}" for p in (ROOT / "configs/robots/arms").glob("*.yaml")),
    "dexengine/record/render/idle.py",
    "dexengine/rl/tasks/scene/object_pin.py",
    "dexengine/record/render/cameras.py",
    "dexengine/record/session/entry.py",
    "dexengine/record/session/runner.py",
    "dexengine/record/render/clip_session.py",
    "dexengine/record/render/clip_phases.py",
    "dexengine/record/looks/look_run.py",
    "dexengine/record/render/writer.py",
    "dexengine/record/rollout/recorder.py",
    "dexengine/record/rollout/oracle_gate.py",
    "dexengine/record/render/slots.py",
    "dexengine/record/render/track.py",
    "dexengine/rl/eval/traj_record.py",
    "dexengine/rl/eval/traj_meta.py",
    # session layout (one session per look), the episode/look plan and the Kit arguments decide what
    # is drawn, so a change there must miss the cache as well
    "dexengine/record/render/clip.py",
    "dexengine/record/render/clip_all.py",
    "dexengine/record/episodes.py",
    "dexengine/record/session/kit.py",
    # visual augmentation (distractors, colours, swept volumes) is drawn into the frames: every module of the package
    *sorted(
        f"dexengine/record/looks/augvis/{p.name}" for p in (Path(__file__).parent / "looks" / "augvis").glob("*.py")
    ),
)


#: replay env code fingerprint: the stage-1 cache and the record-only round record.json keep env_fp; when a stage-1
#:   cache is picked up and the current value differs, only the marker [COLLECT_CACHE] env_diff is printed; it is not
#:   part of the key (behaviour unchanged; the cache is used as is). Old caches without a hash = unknown. Rule (not a
#:   list) = every .py of the replay env package dexengine/rl/tasks (env, mdp/terminations and guards included).
ENV_CODE_GLOB = "dexengine/rl/tasks/**/*.py"
CACHE_MARK = "[COLLECT_CACHE]"


#: file next to the rollouts sha at the stage-1 key place: the record (record.json) content stays as it is (the merge's
#: same-content check compares the record fields; more fields would make old kept keys 'different content'), so the
#: sha goes into one file compared by bytes (record.clean.CARRY).
STAGE1_SHA_FILE = "rollouts.sha256"


def stage1_hit(d: Path) -> bool:
    """When the stage-1 key place d has a record (ok) and the trajectories: check against the sha file and return true
    (a mismatch dies: rewritten after it was built · an old place without the file gets one written) · otherwise false
    (the builder calls stage1_done). One job owns the work place (piece.work_root; sim too, inside the job tree); this
    check is the independent oracle of that rule."""
    rec, npz, sf = d / "record.json", d / "rollouts.npz", d / STAGE1_SHA_FILE
    if not ((json.loads(rec.read_text()) if rec.is_file() else {}).get("ok") and npz.is_file()):
        return False
    got = sha256_file(npz)
    if not sf.is_file():
        _write_text(sf, got + "\n")
        print(f"{CACHE_MARK} [STAGE1_SHA] old place: sha {got[:12]} written · {d.name}", flush=True)
    elif (want := sf.read_text().strip()) != got:
        raise SystemExit(
            f"{CACHE_MARK} ✗ stage1/{d.name} rollouts sha {got[:12]} != built {want[:12]}: rewritten after it was built"
        )
    return True


def stage1_done(d: Path, doc: dict) -> None:
    """A finished stage-1 key place: write the record (content as it is) and the rollouts sha file (checked by
    stage1_hit) atomically."""
    npz = d / "rollouts.npz"
    _write_text(d / "record.json", json.dumps(doc, indent=1))
    if npz.is_file():
        _write_text(d / STAGE1_SHA_FILE, sha256_file(npz) + "\n")


def stage1_put(src: Path, d: Path) -> bool:  # noqa: V103 — public API for plugins (seed reuse)
    """Trajectory file src to the stage-1 key place d/rollouts.npz, written once and never changed: placed atomically
    and true when missing · false when the bytes are the same · dies on different bytes (the same key is never
    rewritten with other content). Caller = seed writing
    (collect_seed_use.take)."""
    import shutil

    d.mkdir(parents=True, exist_ok=True)
    dst = d / "rollouts.npz"
    if dst.is_file():
        if (have := sha256_file(dst)) != (want := sha256_file(src)):
            raise SystemExit(
                f"{CACHE_MARK} ✗ stage1/{d.name} rollouts sha {have[:12]} != to place {want[:12]}: the "
                "same key is never rewritten with other content"
            )
        return False
    t = dst.with_name(f"rollouts.npz.tmp{os.getpid()}")
    shutil.copyfile(src, t)
    os.replace(t, dst)
    return True


def _write_text(p: Path, text: str) -> None:
    t = p.with_name(p.name + f".tmp{os.getpid()}")
    t.write_text(text)
    os.replace(t, p)


def code_fingerprint(root: Path = ROOT, files: tuple[str, ...] = CODE_FILES) -> str:
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return h.hexdigest()[:16]


def render_fingerprint(tree: Path) -> str:
    """Stage-2 fingerprint = render code bytes + the tail of switches that change the render outputs
    (chain.render_tag; off = no tail). The desk adds no tail: one desk (Willow) for every scene — the old Willow
    fingerprint form (render_unify removed the box desk's +desk_box_ tail)."""
    return code_fingerprint(tree, RENDER_CODE_FILES) + CANON.render_tag() + PM.tag()


def cache_key(tree_clips: list[str], ckpt_sha: str, rollouts: int, code_fp: str, partner: bool = False) -> str:
    """Force partner decomposition on and the kept trajectory count (K_TOP) are part of the key (off = old key): records
    of other conditions are not reused."""
    s = json.dumps(
        dict(
            clips=sorted(tree_clips),
            ckpt=ckpt_sha,
            rollouts=int(rollouts),
            code=code_fp,
            k_top=int(REC.K_TOP),
            **({"partner": True} if partner else {}),
        ),
        sort_keys=True,
    )
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def env_fingerprint(root: Path = ROOT) -> str:
    """Replay env code fingerprint: bytes of the ENV_CODE_GLOB files (sorted relative paths). 0 files dies."""
    files = tuple(sorted(p.relative_to(root).as_posix() for p in root.glob(ENV_CODE_GLOB)))
    if not files:
        raise SystemExit(f"{CACHE_MARK} ✗ env code glob {ENV_CODE_GLOB} matched 0 files under {root}")
    return code_fingerprint(root, files)


def env_check(recorded: str | None, now: str, where: str) -> str:
    """The env marker line when a cache is picked up: env_same · env_diff · env_hash unknown (old cache). Behaviour does
    not change."""
    if not recorded:
        return f"{CACHE_MARK} env_hash unknown {where} · now {now}: old cache without a hash (used)"
    if recorded != now:
        return f"{CACHE_MARK} env_diff {where} · recorded {recorded} != now {now}: used as is (marker only)"
    return f"{CACHE_MARK} env_same {where} · {now}"


def cache_line(rec: Path | None, key: str, n: int, level: int, tree: Path) -> None:
    """Stage-1 cache marker: hit (rec = that record.json; the env marker is printed too) · miss (rec None)."""
    print(f"{CACHE_MARK} {'hit' if rec else 'miss'} {key} · clips {n} · rollouts {level}", flush=True)
    if rec:
        print(env_check(json.loads(rec.read_text()).get("env_fp"), env_fingerprint(tree), f"stage1/{key}"), flush=True)
