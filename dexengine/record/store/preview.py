"""Human preview of a stored dataset (CPU, retarget venv) at <out>/preview: format tree, selection table, still frames
(JPEG), video copies or downscales, depth colour-map videos, force curves and manifest.json. Names, size caps and
episode reading belong to store.store; this module only draws. Last line: [COLLECT_PREVIEW] DONE.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from dexengine.record.render import writer as CD
from dexengine.record.rollout import recorder as REC
from dexengine.record.store import force as CF
from dexengine.record.store import store as ST
from dexengine.record.store.parquet import read_table


def format_tree(root: Path) -> list[dict]:
    """The stored layout: files, sizes, parquet columns, npz keys and shapes, mp4 probes."""
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        r = {"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
        if p.suffix == ".parquet":
            t = read_table(p)
            r["rows"] = t.num_rows
            r["columns"] = {
                n: (str(t.schema.field(n).type), list(np.asarray(t.column(n)[0].as_py()).shape)) for n in t.column_names
            }
        elif p.suffix == ".npz":
            z = np.load(p)
            r["keys"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
        elif p.suffix == ".mp4":
            r["video"] = ST._ffprobe(p)
        rows.append(r)
    return rows


def _shrink(src: Path, dst: Path) -> dict:
    """Preview mp4: copy when <= PREVIEW_MP4_MAX_MB, else re-encode at 960 px width, CRF 28 (H.264)."""
    if src.stat().st_size <= ST.PREVIEW_MP4_MAX_MB * 1e6:
        shutil.copy2(src, dst)
        return {"reencoded": False}
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            "scale=960:-2",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ],
        check=True,
    )
    return {"reencoded": True, "rule": "960 px · CRF 28"}


def _stills(src: Path, idx: list[int]) -> dict[int, np.ndarray]:
    """RGB frames idx of an mp4: the same pixels as imageio.mimread(src)[j] (the same ffmpeg reader, rgb24) without
    holding the whole video: ffmpeg's select filter passes only those frames on to the RGB conversion and the pipe.
    A frame the video does not have dies (as mimread(src)[j] did)."""
    import imageio_ffmpeg

    want = sorted(set(idx))
    sel = "+".join(f"eq(n\\,{j})" for j in want)
    gen = imageio_ffmpeg.read_frames(
        str(src),
        pix_fmt="rgb24",  # passthrough: no frames repeated into the gaps
        output_params=["-vf", f"select={sel}", "-fps_mode", "passthrough"],
    )
    w, h = next(gen)["size"]
    got = [np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in gen]
    if len(got) != len(want):
        raise IndexError(f"[COLLECT_PREVIEW] {src}: frames {want} asked, the video gave {len(got)}")
    return dict(zip(want, got, strict=True))


def _jpg(img, path: Path) -> None:
    """Preview still: JPEG (q 90) downscaled to width ST.PREVIEW_IMG_W."""
    from PIL import Image

    im = Image.fromarray(np.asarray(img)[..., :3])
    if im.width > ST.PREVIEW_IMG_W:
        im = im.resize((ST.PREVIEW_IMG_W, round(im.height * ST.PREVIEW_IMG_W / im.width)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)


def _video_previews(rd, picks, tag, pv, add, it):
    """Stills (start, mid, end) and a downscaled copy of each stored video of one episode."""
    for v, key in ST.VIDEO_KEYS.items():
        frames = _stills(rd / f"{v}.mp4", list(picks.values()))
        for lab, j in picks.items():
            rel = f"frames/{tag}_{key.split('.')[-1]}_{lab}.jpg"
            _jpg(frames[j], pv / rel)
            add(rel, f"{it['clip']} · {key} · video frame {j} ({lab})")
        (pv / ST.PREVIEW_VIDEO_DIR).mkdir(exist_ok=True)
        rel = f"{ST.PREVIEW_VIDEO_DIR}/{tag}_{key.split('.')[-1]}.mp4"
        how = _shrink(rd / f"{v}.mp4", pv / rel)
        note = f" (preview downscale {how['rule']})" if how["reencoded"] else ""
        add(rel, f"{it['clip']} · {key} · 30 fps H.264{note}")


def _depth_previews(picks, tag, dz, pv, add, it):
    """Left-eye depth: colour-map stills and a downscaled colour-map video."""
    import imageio.v2 as imageio

    from dexengine.record.render import cameras as CC

    for lab, j in picks.items():
        rel = f"frames/{tag}_depth_ego_left_{lab}.jpg"
        _jpg(CC.depth_colormap(dz[j] / 1000.0), pv / rel)
        add(
            rel,
            f"{it['clip']} · left-eye depth colour map (turbo · {CC.DEPTH_VIZ_RANGE_M} m · black = invalid) frame {j}",
        )
    rel = f"{ST.PREVIEW_VIDEO_DIR}/{tag}_depth_ego_left.mp4"
    tmp = pv / ST.PREVIEW_VIDEO_DIR / f"_{tag}_depth_full.mp4"
    imageio.mimwrite(
        tmp,
        [CC.depth_colormap(d / 1000.0) for d in dz],
        fps=30,
        codec="libx264",
        quality=None,
        ffmpeg_params=["-crf", "23"],
        macro_block_size=8,
    )
    how = _shrink(tmp, pv / rel)
    tmp.unlink()
    add(rel, f"{it['clip']} · left-eye depth colour-map video 30 fps{' (downscaled)' if how['reencoded'] else ''}")


def _curves_preview(curve_hz, ep, it, tag, pv, add):
    """Downsampled curves (joints, policy output, tracking, per-sensor force sum) of one episode."""
    k = max(1, int(round(1.0 / (curve_hz * ep["step_dt"]))))
    c = ep["cols"]
    fo, ft = CF.force_timeline(ep["forces"], ep["step_dt"], ep["physics_dt"], ep["L"])
    kf = max(1, int(round(1.0 / (curve_hz * ep["physics_dt"]))))
    curves = {
        "clip": it["clip"],
        "hz": curve_hz,
        "t": c["timestamp"][::k].round(3).tolist(),
        "series": {
            name: np.asarray(v)[::k].round(5).tolist()
            for name, v in c.items()
            if name.startswith(("observation.joint_pos", "action", "tracking"))
        },
        "names": {ch["name"]: ch["names"] for ch in ep["channels"]},
        "force_t": ft[::kf].round(3).tolist() if ft is not None else [],
        "force_norm_n": {
            s: np.linalg.norm(v["force_n"], axis=-1).sum(-1)[::kf].round(4).tolist()
            for s, v in fo.items()
            if s.split(".", 1)[0] in REC.FORCE_KINDS
        },
        "tracking_summary": ep["errors"],
    }
    rel = f"curves_{tag}.json"
    (pv / rel).write_text(json.dumps(curves, ensure_ascii=False))
    add(
        rel,
        f"{it['clip']} · curves (downsampled {curve_hz:g} Hz): joints · policy output · absolute joint "
        "targets · tracking error · per-sensor force sum |F|",
    )


def preview(ds: Path, pv: Path, items: list[dict], selection: dict, curve_hz: float = 10.0):
    """Write the preview of dataset ds (items = the stored episodes) to pv."""

    pv.mkdir(parents=True, exist_ok=True)
    man = []

    def add(rel, desc):
        man.append({"file": rel, "bytes": (pv / rel).stat().st_size, "desc": desc})

    TF, TS = ST.eval_modules()
    for i, it in enumerate(items):
        tag = f"ep{i}"
        tf = TF.load(it["traj"])
        ep, _rmeta, _extra = ST.rendered_episode(tf, TS, it)
        rd = Path(it["render_dir"])
        dz = CD.read(rd / CD.DEPTH_FILE)
        n = dz.shape[0]
        picks = {"start": 0, "mid": n // 2, "end": n - 1}
        _video_previews(rd, picks, tag, pv, add, it)
        _depth_previews(picks, tag, dz, pv, add, it)
        _curves_preview(curve_hz, ep, it, tag, pv, add)
    (pv / "selection.json").write_text(json.dumps(selection, indent=1, ensure_ascii=False, default=float))
    add(
        "selection.json",
        "selection table: every rollout per clip (success · length · score = traj_select.score) · "
        "top ranks (rollout.recorder.TOP_N) · the one kept trajectory",
    )
    tree = format_tree(ds)
    (pv / "format_tree.json").write_text(json.dumps(tree, indent=1, ensure_ascii=False, default=float))
    add(
        "format_tree.json",
        "stored layout: dataset files · sizes · parquet columns (type · shape) · npz keys · "
        "mp4 (codec · resolution · frames)",
    )
    (pv / "manifest.json").write_text(json.dumps({"dataset": str(ds), "files": man}, indent=1, ensure_ascii=False))
    big = [m["file"] for m in man if m["file"].endswith(".mp4") and m["bytes"] > ST.PREVIEW_MP4_MAX_MB * 1e6]
    print(f"[COLLECT_PREVIEW] DONE files {len(man)} · mp4 > {ST.PREVIEW_MP4_MAX_MB} MB {big} · {pv}", flush=True)
    if big:
        raise SystemExit("[COLLECT_PREVIEW] preview mp4 over the size cap")
