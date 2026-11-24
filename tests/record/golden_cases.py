"""Deterministic record functions run on small synthetic inputs; tests/record/golden/cases.json holds their recorded
results, and tests/record/test_golden.py compares this tree against them.

run(resolve) -> {case id: result}; resolve(key) returns the module for one of the keys in MODULES. A result is the
function's return value made JSON-safe, or {"raises": <exception type>}.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from experiments.workspace import LEGACY_ENV_PREFIX

#: key -> module of this tree
MODULES = {
    "lerobot": "dexengine.record.store.lerobot",
    "form": "dexengine.record.store.form",
    "merge_check": "dexengine.record.store.merge_check",
    "merge_rows": "dexengine.record.store.merge_rows",
    "rtx": "dexengine.record.looks.materials",
    "rev": "dexengine.record.looks.rev",
    "look": "dexengine.record.looks.look_d",
    "depth": "dexengine.record.render.writer",
    "force": "dexengine.record.store.force",
    "headview": "dexengine.record.render.visibility",
    "meta": "dexengine.record.store.meta",
    "canon": "dexengine.record.chain",
}


def _safe(x):
    if isinstance(x, np.ndarray):
        return {"ndarray": x.tolist(), "dtype": str(x.dtype)}
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v) for v in x]
    if isinstance(x, set):
        return sorted(_safe(v) for v in x)
    return x


def _outcome(fn):
    """One case's value, or the name of the exception it raises (the golden records both)."""
    try:
        with redirect_stdout(io.StringIO()):
            return _safe(fn())
    except Exception as e:  # noqa: BLE001 — the golden records whichever exception a case raises
        return {"raises": type(e).__name__}


def _info(video_keys=("ego_left", "exo"), extra=None):
    feats = {
        "observation.state": {"dtype": "float32", "shape": [3], "names": ["a", "b", "c"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
        **{
            f"observation.images.{k}": {"dtype": "video", "shape": [4, 6, 3], "names": ["h", "w", "c"]}
            for k in video_keys
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "rby1_test",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 10,
        "total_tasks": 1,
        "total_videos": 2 * len(video_keys),
        "total_chunks": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feats,
        "collect": {
            "video_keys": [f"observation.images.{k}" for k in video_keys],
            "depth_path": "depth/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npy.zst",
        },
    }
    if extra:
        info["collect"].update(extra)
    return info


def _ep(i, clip, rnd, look="D", src=None, row0=None, row0_source=None, piece="p1"):
    tc = {}
    if row0 is not None:
        tc = {"row0": row0, "orig_fov": {"rule": "orig_fov_out_row0_fill", "row0_source": row0_source}}
    return {
        "episode_index": i,
        "clip": clip,
        "round": rnd,
        "look": {"name": look},
        "piece_id": piece,
        "src_episode_index": i,
        "selection": {"source_clip": src} if src else {},
        "traj_choice": tc,
        "length": 5,
        "tasks": ["t"],
    }


def _switches(C):
    """switches() / render_tag() with the aug placement switch set explicitly (env names normalized)."""
    import os

    env = C.AUGC.ENV
    old = os.environ.get(env)
    os.environ[env] = "1"
    try:
        sw = C.switches()
        tag = C.render_tag()
    finally:
        if old is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = old
    return json.loads(json.dumps(dict(sw, render_tag=tag)).replace(LEGACY_ENV_PREFIX, "DEXENGINE_"))


def cases(resolve):
    L, F, M, X, V = (resolve(k) for k in ("lerobot", "form", "merge_check", "rtx", "rev"))
    MR = resolve("merge_rows")
    rule = getattr(MR.EPI, "ORIG_FOV_RULE", None)
    aug = [
        _ep(0, "c1", "orig"),
        _ep(1, "c1_pl_v1", "aug", src="c1", row0="c1_pl_v1", row0_source="variant"),
        _ep(2, "c2", "aug", src=None, row0="c2", row0_source="orig"),
    ]
    for e in aug:
        if e["traj_choice"]:
            e["traj_choice"]["orig_fov"]["rule"] = rule
    cur = [_ep(0, "c1", "orig"), _ep(1, "c2", "aug", row0="c2", row0_source="orig")]
    cur[1]["traj_choice"]["orig_fov"]["rule"] = rule
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(8, 10, 3), dtype=np.uint8)
    out = {
        "lerobot.union_order": lambda: L.union_order([["a", "b", "c"], ["a", "c", "d"], ["b", "e"]]),
        "lerobot.sample_indices": lambda: [L.sample_indices(n) for n in (1, 5, 37, 200, 1000)],
        "lerobot.totals": lambda: L.totals(10, 1234, 3, 1000, 3),
        "lerobot.parquet_columns": lambda: L.parquet_columns(_info()),
        "form.fingerprint": lambda: F.fingerprint(_info()),
        "form.form_diff.same": lambda: F.form_diff(_info(), _info()),
        "form.rules": lambda: [F.RESTING_RULE, F.FILL_RULE],
        "form.stats_skip": lambda: F.stats_skip(_ep(0, "c1", "orig")),
        "merge_check.shape_diff.same": lambda: M.shape_diff(_info(), _info()),
        "merge_check.shape_diff.videos": lambda: M.shape_diff(_info(), _info(video_keys=("ego_left",))),
        "merge_check.row0_clips": lambda: MR.row0_clips(aug),
        "merge_check.old_tree_aug": lambda: MR.old_tree_aug(aug + [_ep(9, "c9", "aug")]),
        "merge_check.variant_row0_clips": lambda: MR.variant_row0_clips(aug),
        "merge_check.fallback_row0": lambda: [MR.fallback_row0(e) for e in aug + cur],
        "merge_check.supersede_plan": lambda: MR.supersede_plan(cur, set(), aug, "p2"),
        "merge_check.aug_row0_plan": lambda: MR.aug_row0_plan(
            cur,
            set(),
            [_ep(5, "c2_pl_v3", "aug", src="c2", row0="c2_pl_v3", row0_source="variant")],
            "p3",
            {("aug", "c2")},
        ),
        "merge_check.families": lambda: M.families(_info()),
        "merge_check.card": lambda: [MR.card(e) for e in aug],
        "merge_check.row0_source": lambda: [MR.row0_source(e, {2}) for e in aug],
        "merge_check.piece_meta_sha": lambda: M.piece_meta_sha(lambda rel: hashlib.sha256(rel.encode()).digest()),
        "rtx.blackbody_rgb": lambda: [X.blackbody_rgb(k) for k in (1800, 2222, 3000, 4000, 4000.5, 5000, 6500, 9000)],
        "rev.probe_frame": lambda: [V.probe_frame(g, 100, 30.0) for g in ({"onset_s": 1.2}, {}, {"onset_s": 9.0})],
        "rev.probe_luma": lambda: V.probe_luma(rgb),
        "rev.d_post": lambda: V.d_post(rgb, "probe", None),
        "rev.d_post.dark": lambda: V.d_post(rgb // 4, "probe", 1.5),
        "rev.r_post": lambda: V.r_post(rgb, "probe"),
        "rev.r_post.dark": lambda: V.r_post(rgb // 4, "probe"),
    }

    def _mod(k):
        try:
            return resolve(k)
        except Exception as e:  # noqa: BLE001 — any import failure becomes one RuntimeError naming the module key
            raise RuntimeError(f"import {k}: {type(e).__name__}") from e

    out.update(
        {
            # constants and builders whose values land in dataset metadata (merge compares them)
            "look.meta": lambda: _mod("look").meta(),
            "depth.CODEC": lambda: _mod("depth").CODEC,
            "force.RULE": lambda: [_mod("force").RULE, _mod("force").QUALITY_KEYS],
            "headview.rule": lambda: [
                _mod("headview").RULE,
                _mod("headview").info_fields([{"head_view": {}}]),
                _mod("headview").info_fields([{}]),
            ],
            "meta.loader": lambda: [_mod("meta").loader_meta(30), _mod("meta").CAMERA_FRAME, _mod("meta").CAMERA_AXES],
            "canon.switches": lambda: _switches(_mod("canon")),
        }
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tail.bin"
        p.write_bytes(bytes(range(256)) * 600)
        out["merge_check.tail_sha"] = lambda: [M.tail_sha(p, n) for n in (10, 1000, p.stat().st_size)]
        res = {k: _outcome(fn) for k, fn in out.items()}
    return json.loads(json.dumps(res, default=str))
