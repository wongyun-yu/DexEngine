"""Store metadata checks: camera values carried from the render meta,
the depth hard-link placement, and the fresh output folder."""

from __future__ import annotations

import ast
import inspect
import json
import os

import pytest

from dexengine.record.render import writer as CW
from dexengine.record.store import meta as CM
from dexengine.record.store import store as ST


def _rmeta_cam(fx: float = 732.0, delta: float = 0.12) -> dict:
    zed = {
        "width": 1280,
        "height": 720,
        "fx": fx,
        "fy": fx,
        "cx": 633.6,
        "cy": 356.2,
        "distortion": [0.0] * 5,
        "fps": 30,
        "baseline_m": 0.0629,
        "mode": "HD720",
        "render_width": 1292,
        "crop_x0": 12,
    }
    return {
        "fps": 30,
        "camera": {
            "zed": zed,
            "K_left": [[fx, 0.0, 633.6], [0.0, fx, 356.2], [0.0, 0.0, 1.0]],
            "K_exo": [[600.0, 0, 640], [0, 600.0, 360], [0, 0, 1]],
            "exo_res": [1280, 720],
            "ego_left_pose": {"pos": [0.1, 0.03, 1.4], "quat_ros_wxyz": [0.5, -0.5, 0.5, -0.5], "fwd": [1, 0, 0]},
            "ego_right_pose": {"pos": [0.1, -0.03, 1.4], "quat_ros_wxyz": [0.5, -0.5, 0.5, -0.5]},
            "exo": {
                "pos": [1.5, 0.0, 1.2],
                "quat_opengl_wxyz": [0.0, 0.0, 0.7071, 0.7071],
                "canon_pos": [1.38, 0.0, 1.2],
                "delta_m": delta,
                "margin_frac": 0.05,
                "upper_links": ["torso"],
            },
            "head_joint": "head_1",
            "head_pinned_deg": {"head_0": 0.0, "head_1": 31.5},
            "framing": {"rule": "desk"},
        },
    }


def test_camera_values_are_the_render_meta_values():
    rm = _rmeta_cam()
    cam = rm["camera"]
    ki, kp = CM.camera_intrinsics(rm), CM.camera_pose(rm)
    assert ki["ego"]["K"] == cam["K_left"] and ki["exo"]["K"] == cam["K_exo"] and ki["ego"]["baseline_m"] == 0.0629
    assert ki["ego"]["distortion"] == cam["zed"]["distortion"]
    assert (ki["exo"]["width"], ki["exo"]["height"]) == tuple(cam["exo_res"])
    assert ki["ego"]["fps"] == 30 and ki["video_fps"] == 30 and "render_width" not in ki["ego"]
    assert kp["ego_left"] == {k: cam["ego_left_pose"][k] for k in ("pos", "quat_ros_wxyz")}
    assert kp["exo"]["delta_m"] == 0.12 and kp["exo"]["quat_opengl_wxyz"] == cam["exo"]["quat_opengl_wxyz"]
    assert kp["head_pinned_deg"] == cam["head_pinned_deg"]


@pytest.mark.parametrize("cut", [("camera",), ("camera", "exo", "delta_m"), ("camera", "zed", "fx")])
def test_camera_values_missing_field_dies(cut):
    rm = json.loads(json.dumps(_rmeta_cam()))
    d = rm
    for k in cut[:-1]:
        d = d[k]
    del d[cut[-1]]
    with pytest.raises((SystemExit, KeyError, ValueError)):
        if cut[-1] != "fx":
            CM.camera_pose(rm)
        else:
            CM.camera_intrinsics(rm)


def test_cameras_info_one_intrinsics_and_streams():
    cam = _rmeta_cam()["camera"]
    one = ST.cameras_info([CM.camera_intrinsics(_rmeta_cam(delta=0.1)), CM.camera_intrinsics(_rmeta_cam(delta=0.3))])
    assert one["ego"]["K"] == cam["K_left"] and one["streams"][ST.DEPTH_KEY].startswith("ego")
    assert set(one["streams"]) == {*ST.VIDEO_KEYS.values(), ST.DEPTH_KEY}
    with pytest.raises(SystemExit):
        ST.cameras_info([CM.camera_intrinsics(_rmeta_cam()), CM.camera_intrinsics(_rmeta_cam(fx=700.0))])


def test_render_fix_carries_camera_pose():
    fn = next(
        f
        for f in ast.walk(ast.parse(inspect.getsource(CM)))
        if isinstance(f, ast.FunctionDef) and f.name == "render_fix"
    )
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "camera_pose" for n in ast.walk(fn))


def test_depth_place_hard_links_same_content(tmp_path):
    a, b = tmp_path / "a.npy.zst", tmp_path / "b.npy.zst"
    a.write_bytes(b"same-depth-bytes" * 64)
    b.write_bytes(b"other-depth-byte" * 64)
    d = tmp_path / "ds"
    r0 = CW.place(a, d / "episode_000000.npy.zst")
    r1 = CW.place(a, d / "episode_000001.npy.zst")
    r2 = CW.place(b, d / "episode_000002.npy.zst")
    ino = [os.stat(d / f"episode_00000{k}.npy.zst").st_ino for k in range(3)]
    assert r0 is None and r1 == "episode_000000.npy.zst" and ino[0] == ino[1]
    assert r2 is None and ino[2] != ino[0]
    CW.place(a, d / "episode_000001.npy.zst")  # rerun: dst is already a link of the same file
    assert sorted(q.name for q in d.iterdir() if q.name.endswith(".tmp")) == []
    CW.place(b, d / "episode_000001.npy.zst")  # redraw: a new entry, the linked sibling keeps its content
    assert (d / "episode_000000.npy.zst").read_bytes() == a.read_bytes()
    assert (d / "episode_000001.npy.zst").read_bytes() == b.read_bytes()


def test_fresh_out_clears_an_old_run(tmp_path):
    out = tmp_path / "rounds/orig/dataset"
    (out / "data/chunk-000").mkdir(parents=True)
    for i in range(5, 10):
        (out / f"data/chunk-000/episode_{i:06d}.parquet").write_bytes(b"old")
    (out / "meta").mkdir()
    (out / "meta/info.json").write_text("{}")
    assert ST.fresh_out(out) == 5 and out.is_dir() and not any(out.iterdir())
    assert ST.fresh_out(out) == 0 and out.is_dir()
