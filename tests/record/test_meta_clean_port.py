"""Head joints from the render measurement (store.meta.render_head)
and the bundle cleanup that runs only after verification (record.clean.clean)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dexengine.record import clean as CLN
from dexengine.record.render import episode as CRN
from dexengine.record.store import meta as CM


def _head_case():
    names = ["head_0", "arm_0", "head_1"]
    ents = [
        {"name": "robot", "kind": "artic", "joint_names": names},
        {"name": "object_right", "kind": "rigid", "joint_names": []},
    ]
    L = 7
    st = np.zeros((L, 3), np.float32)
    st[:, 0] = np.linspace(0, -0.001, L).astype(np.float32)  # head_0 follows the trajectory
    st[:, 1] = 0.3
    st[:, 2] = np.float32(0.5236)  # head_1: 30 deg in the rollout
    tg = np.tile(np.array([0.0, 0.3, 0.5236], np.float32), (L, 1))
    cols = {"observation.joint_pos.robot": st, "action.joint_target.robot": tg}
    chans = [{"kind": "joint_target", "name": "joint_target/robot", "names": names}]
    steps = [0, 2, 4, 6]
    v = float(np.float32(0.79))  # render auto pitch (float32 measurement)
    rm = {
        "frame_step": steps,
        "head": {
            "joints": ["head_0", "head_1"],
            "pinned_rad": {"head_1": 0.79},
            "frame_rad": [[float(st[t, 0]), v] for t in steps],
        },
    }
    return cols, ents, chans, rm, v


def test_render_head_replaces_the_pinned_joint_only():
    cols, ents, chans, rm, v = _head_case()
    st0, tg0 = cols["observation.joint_pos.robot"].copy(), cols["action.joint_target.robot"].copy()
    keep_in = cols["observation.joint_pos.robot"]
    out = CM.render_head(cols, ents, chans, rm)
    s, t = cols["observation.joint_pos.robot"], cols["action.joint_target.robot"]
    assert np.all(s[:, 2] == np.float32(v)) and np.all(t[:, 2] == np.float32(v))
    assert np.array_equal(s[:, :2], st0[:, :2]) and np.array_equal(t[:, :2], tg0[:, :2])
    assert np.array_equal(keep_in, st0)
    assert out["head_1"]["source"] == "render_pinned" and abs(out["head_1"]["rollout_state_deg"][0] - 30.0) < 1e-3
    assert out["head_0"]["source"] == "trajectory" and out["head_0"]["frame_max_diff_rad"] == 0.0


@pytest.mark.parametrize("bad", ["pinned_moves", "traj_differs", "no_head", "two_heads"])
def test_render_head_negatives_die(bad):
    c, e, ch, r, _ = _head_case()
    if bad == "pinned_moves":
        r["head"]["frame_rad"][2][1] += 1e-4
    elif bad == "traj_differs":
        r["head"]["frame_rad"][1][0] += 1e-4
    elif bad == "no_head":
        r = {k: x for k, x in r.items() if k != "head"}
    else:
        e = e + [dict(e[0], name="robot2")]
    with pytest.raises((SystemExit, ValueError, KeyError)):
        CM.render_head(c, e, ch, r)


def _bundle(root: Path) -> tuple[Path, Path]:
    out, ext = root / "bundle", root / "outside"
    ext.mkdir()
    (ext / "keep.bin").write_bytes(b"x" * 10)
    files = {
        "stage1/k1/rollouts.npz": 100,
        "stage1/k1/record.json": 5,
        "stage2/k1/c1/render_meta.json": 7,
        "stage2/k1/c1/session.log": 9,
        "stage2/k1/c1/render_code.txt": 3,
        "preview/frames/ep0_exo_start.jpg": 20,
        "preview/curves_ep0.json": 6,
        "preview/videos/ep0_exo.mp4": 50,
        "preview/videos/ep0_depth_ego_left.mp4": 40,
        "dataset/meta/info.json": 4,
    }
    files.update({f"stage2/k1/c1/{m}": 30 for m in CRN.MEDIA})
    for rel, n in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(b'{"a":1}' if rel.endswith("render_meta.json") else b"a" * n)
    (out / "seq1/c1").mkdir(parents=True)
    (out / "seq1/c1/c1").symlink_to(ext)
    man = [
        {"file": f, "bytes": 1, "desc": ""}
        for f in ("frames/ep0_exo_start.jpg", "videos/ep0_exo.mp4", "videos/ep0_depth_ego_left.mp4", "curves_ep0.json")
    ]
    (out / "preview/manifest.json").write_text(json.dumps({"dataset": "d", "files": man}))
    return out, ext


def _listing(out: Path) -> dict:
    return {
        str(p.relative_to(out)): (p.lstat().st_size if p.is_symlink() else p.stat().st_size)
        for p in out.rglob("*")
        if p.is_file() or p.is_symlink()
    }


def test_clean_only_after_verification(tmp_path):
    out, ext = _bundle(tmp_path)
    before = _listing(out)
    assert not CLN.clean(out, {"ok": False, "episodes": 1}, out, None)["done"] and _listing(out) == before
    r = CLN.clean(out, {"ok": True, "episodes": 1}, out, None)
    after = _listing(out)
    want = sorted(
        [f"stage2/k1/c1/{m}" for m in CRN.MEDIA]
        + ["preview/videos/ep0_depth_ego_left.mp4", "preview/videos/ep0_exo.mp4"]
    )
    assert sorted(set(before) - set(after)) == want
    assert "stage1/k1/rollouts.npz" in after and "stage2/k1/c1/render_meta.json" in after
    assert r["bytes"] == 30 * len(CRN.MEDIA) + 90 and r["files"] == len(want)
    assert json.loads((out / "cleanup.json").read_text())["bytes"] == r["bytes"]
    m = json.loads((out / "preview/manifest.json").read_text())
    assert sorted(f["file"] for f in m["files"]) == ["curves_ep0.json", "frames/ep0_exo_start.jpg"]
    assert len(m["removed"]) == 2
    assert (ext / "keep.bin").is_file() and (out / "seq1/c1/c1").is_symlink()
    assert not (out / "preview/videos").exists()


def test_clean_refuses_media_link_outside_bundle(tmp_path):
    out, ext = _bundle(tmp_path)
    m0 = out / "stage2/k1/c1" / CRN.MEDIA[0]
    m0.unlink()
    m0.symlink_to(ext / "keep.bin")
    before = _listing(out)
    with pytest.raises(SystemExit):
        CLN.clean(out, {"ok": True, "episodes": 1}, out, None)
    assert _listing(out) == before and (ext / "keep.bin").is_file()


def test_clean_refuses_oversized_preview(tmp_path):
    out, _ext = _bundle(tmp_path)
    (out / "preview/frames/big.jpg").write_bytes(b"b" * int(CLN.PREVIEW_KEEP_MB_PER_EP * 1e6 + 1))
    before = _listing(out)
    with pytest.raises(SystemExit):
        CLN.clean(out, {"ok": True, "episodes": 1}, out, None)
    assert _listing(out) == before
