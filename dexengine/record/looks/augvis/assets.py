#!/usr/bin/env python3
"""Asset receipt for the record render visual augment: Poly Haven (CC0 textures, models, indoor HDRIs), ambientCG (CC0
materials) and Google Scanned Objects (CC BY 4.0 models) are fetched into a local mirror (AUGVIS_ASSETS) and listed.

Default = the canonical generation (sources.ASSET_GEN): its list is kept in git (LIST_DIR = configs/record/<name>/:
manifest.json + SHA256SUMS of the canonical root) and is the input — every id comes from it, every fetched file is
checked against its sha256 row, a file on disk from an earlier fetch is checked the same way; a differing, missing or
unlisted file dies naming it, a missing or other list dies (no live choice). The list's manifest.json is installed as
is: its metadata (dims · categories) and decoded means (mean_srgb · mean_lum · look_d) are those of the canonical fetch,
its stored old values translated to English by the site migration (sources.ASSET_GEN manifest_sha256). A root that
already holds another manifest dies before anything is fetched (the canonical root keeps its original values).
Why: the live choice (pick_* rank by today's download counts and listings) gave another set than the canonical one.
`--new-generation`: today's live choice (augvis.sources pick_acg · pick_gso · pick_hdris · augvis.catalog ph_ids) with
a rebuilt manifest and list — not the canonical generation: the render dies on it until its list is committed at
configs/record/<name>/ and ASSET_GEN names it.
Received: textures = diffuse · rough · normal (GL) jpg · Poly Haven models = usd (usdc + its textures) ·
GSO = OBJ + texture.png -> usda (UsdPreviewSurface network · 1024 jpg texture) ·
HDRI = 2k .hdr (mean luminance, for dome intensity normalisation).
manifest.json: textures[id] · models[id] · hdris[id] · acg[axis] · gso_pick · look_d · src · license · page per asset.
Root = env AUGVIS_ASSETS (required, no silent default). Markers [AUG_VIS_ASSETS] … end line DONE n/n.
Usage: AUGVIS_ASSETS=<folder> <Isaac python, pxr · cv2> dexengine/record/looks/augvis/assets.py
       [--verify | --new-generation]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from dexengine.core import capacity, config, paths
from dexengine.core.errors import InputMissingError
from dexengine.record.looks.augvis import catalog as AC
from dexengine.record.looks.augvis import sources as SR
from dexengine.record.looks.augvis.obj_usd import obj_to_usd

MARK = "[AUG_VIS_ASSETS]"
UA = {"User-Agent": "dexengine-augvis/2.0"}
STEP = "dexengine.record.looks.augvis.assets"  # capacity.NEEDS key (download threads)
#: The list of the canonical generation kept in git: manifest.json + SHA256SUMS of its root (the fetch input).
LIST_DIR = paths.CONFIGS_DIR / "record" / SR.ASSET_GEN["name"]
NOT_ASSETS = ("manifest.json", "SHA256SUMS", "LICENSE.txt")  # files of a root that are not listed in its SHA256SUMS


def root() -> Path:
    r = os.environ.get("AUGVIS_ASSETS", "")
    if not r:
        raise InputMissingError(f"{MARK} env AUGVIS_ASSETS missing (asset root; no default)")
    return Path(r)


def get_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read())


def fetch(url: str, dst: Path, md5: str | None = None) -> None:
    if dst.is_file() and (md5 is None or hashlib.md5(dst.read_bytes()).hexdigest() == md5):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            break
        except Exception as e:  # transient network errors only, 3 times
            if attempt == 2:
                raise SystemExit(f"{MARK} download failed {url}: {e}") from e
            time.sleep(3)
    if md5 and hashlib.md5(tmp.read_bytes()).hexdigest() != md5:
        raise SystemExit(f"{MARK} md5 mismatch {url}")
    tmp.rename(dst)


def mean_srgb(img: Path) -> list[float]:
    from PIL import Image

    im = Image.open(img).convert("RGB").resize((64, 64))
    px = list(im.getdata())
    return [round(sum(p[i] for p in px) / len(px) / 255.0, 4) for i in range(3)]


def look_d_albedo() -> dict:
    """BaseColor mean of the look D materials (NVIDIA Base MDL, look_d mirror): the D reference of the contrast rule."""
    m = paths.main_root() / "data/assets/scene_canon/collect_look_d/Assets/Isaac/4.5/NVIDIA/Materials/Base"
    out = {}
    for k, rel in (
        ("desk", "Wood/Birch/Birch_BaseColor.png"),
        ("floor", "Masonry/Concrete_Polished/Concrete_Polished_BaseColor.png"),
        ("wall", "Wall_Board/Plaster/Plaster_BaseColor.png"),
    ):
        p = m / rel
        if not p.is_file():
            raise InputMissingError(f"{MARK} D material texture {p} missing (look_d mirror)")
        out[k] = mean_srgb(p)
    return out


def get_bytes(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: PERF203 — retry loop: transient network errors only, 3 times
            if attempt == 2:
                raise SystemExit(f"{MARK} download failed {url}: {e}") from e
            time.sleep(3)
    raise AssertionError


def ph_fetch(r: Path, kind: str, pid: str, res: str) -> dict:
    """One Poly Haven texture (diffuse · rough · normal GL jpg) or model (usd + its includes) at res -> relative paths
    ({map: path} | {"usd": path})."""
    files = get_json(SR.API_FILES.format(id=pid))
    if kind == "texture":
        out = {}
        for mk, sub in (("diff", "Diffuse"), ("rough", "Rough"), ("nor_gl", "nor_gl")):
            ent = files[sub][res]["jpg"]
            dst = r / "textures" / pid / Path(ent["url"]).name
            fetch(ent["url"], dst, ent.get("md5"))
            out[mk] = str(dst.relative_to(r))
        return out
    ent = files["usd"][res]["usd"]
    d = r / "models" / pid
    usd = d / Path(ent["url"]).name
    fetch(ent["url"], usd, ent.get("md5"))
    for rel, inc in ent.get("include", {}).items():
        fetch(inc["url"], d / rel, inc.get("md5"))
    return {"usd": str(usd.relative_to(r))}


def hdri_fetch(r: Path, pid: str) -> Path:
    """One Poly Haven HDRI (.hdr at HDRI_RES)."""
    ent = get_json(SR.API_FILES.format(id=pid))["hdri"][SR.HDRI_RES]["hdr"]
    dst = r / "hdris" / Path(ent["url"]).name
    fetch(ent["url"], dst, ent.get("md5"))
    return dst


def acg_fetch(r: Path, aid: str) -> dict:
    """One ambientCG material (1K-JPG zip: Color · Roughness · NormalGL) -> {map: relative path}; no Color map dies."""
    d = r / "textures" / aid
    if not (d / f"{aid}_Color.jpg").is_file():
        z = zipfile.ZipFile(io.BytesIO(get_bytes(SR.ACG_ZIP.format(id=aid))))
        d.mkdir(parents=True, exist_ok=True)
        for n in z.namelist():
            for k in ("Color", "Roughness", "NormalGL"):
                if n.endswith(f"_{k}.jpg"):
                    (d / f"{aid}_{k}.jpg").write_bytes(z.read(n))
    files = {}
    for mk, k in (("diff", "Color"), ("rough", "Roughness"), ("nor_gl", "NormalGL")):
        if (d / f"{aid}_{k}.jpg").is_file():
            files[mk] = str((d / f"{aid}_{k}.jpg").relative_to(r))
    if "diff" not in files:
        raise InputMissingError(f"{MARK} ambientCG {aid} has no Color map")
    return files


def gso_fetch(r: Path, name: str) -> list[float]:
    """One Google Scanned Object (zip: OBJ + texture.png) -> model.usda + texture.jpg (1024) + dims.json; dims (m)."""
    from PIL import Image

    d = r / "models" / "gso" / name
    usd, tex = d / "model.usda", d / "texture.jpg"
    if not (usd.is_file() and tex.is_file()):
        z = zipfile.ZipFile(io.BytesIO(get_bytes(SR.GSO_ZIP.format(name=name))))
        d.mkdir(parents=True, exist_ok=True)
        obj = d / "model.obj"
        obj.write_bytes(z.read("meshes/model.obj"))
        im = Image.open(io.BytesIO(z.read("materials/textures/texture.png"))).convert("RGB")
        im.resize((1024, 1024), Image.LANCZOS).save(tex, quality=90)
        dims = obj_to_usd(obj, tex, usd)
        obj.unlink()
        (d / "dims.json").write_text(json.dumps(dims))
    return json.loads((d / "dims.json").read_text())


def polyhaven(r: Path, man: dict) -> None:
    allinfo = get_json("https://api.polyhaven.com/assets")
    src = {"src": "polyhaven", "license": SR.SOURCES["polyhaven"]["license"]}
    for kind, pid, res in AC.ph_ids():
        got = ph_fetch(r, kind, pid, res)
        info = allinfo.get(pid, {})
        if kind == "texture":
            rec = dict(src, res=res, dims_mm=info.get("dimensions"), page=f"https://polyhaven.com/a/{pid}", files=got)
            rec["mean_srgb"] = mean_srgb(r / rec["files"]["diff"])
            man["textures"][pid] = rec
        else:
            dims = info.get("dimensions") or [0, 0, 0]
            rec = dict(src, res=res, usd=got["usd"], dims_mm=dims, page=f"https://polyhaven.com/a/{pid}")
            why = AC.size_why([x / 1000.0 for x in dims])
            if why:
                rec["excluded"] = why
            man["models"][pid] = rec
        print(f"{MARK} polyhaven {kind} {pid} {res}", flush=True)


def hdris(r: Path, man: dict) -> None:
    import cv2
    import numpy as np

    names = SR.pick_hdris(get_json(SR.PH_HDRI_API))

    def one(pid):
        dst = hdri_fetch(r, pid)
        img = cv2.imread(str(dst), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float64)
        w = np.cos((np.arange(img.shape[0]) + 0.5) / img.shape[0] * np.pi - np.pi / 2)[:, None]  # latitude area weight
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        return pid, {
            "src": "polyhaven",
            "license": SR.SOURCES["polyhaven"]["license"],
            "res": SR.HDRI_RES,
            "file": str(dst.relative_to(r)),
            "mean_lum": round(float((lum * w).sum() / (w.sum() * img.shape[1])), 5),
            "page": f"https://polyhaven.com/a/{pid}",
        }

    with capacity.sized(STEP, len(names)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        for pid, rec in ex.map(one, names):
            man["hdris"][pid] = rec
            print(f"{MARK} hdri {pid} mean luminance {rec['mean_lum']}", flush=True)


def ambientcg(r: Path, man: dict) -> None:
    assets = get_json(SR.ACG_API, timeout=300)["foundAssets"]
    by_id = {a["assetId"]: a for a in assets}
    pick = SR.pick_acg(assets)
    ids = sorted({i for v in pick.values() for i, _c in v})

    def one(aid):
        files = acg_fetch(r, aid)
        a = by_id[aid]
        dx, dy = float(a.get("dimensionX") or 0), float(a.get("dimensionY") or 0)
        return aid, {
            "src": "ambientcg",
            "license": SR.SOURCES["ambientcg"]["license"],
            "res": "1K-JPG",
            "files": files,
            "dims_mm": [dx * 10, dy * 10] if dx > 0 and dy > 0 else None,
            "dims_from": "dimensionX/Y cm" if dx > 0 else "none — 1 m",  # data (manifest)
            "cat": a.get("displayCategory"),
            "page": f"https://ambientcg.com/a/{aid}",
            "mean_srgb": mean_srgb(r / files["diff"]),
        }

    with capacity.sized(STEP, len(ids)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        for aid, rec in ex.map(one, ids):
            man["textures"][aid] = rec
            print(f"{MARK} ambientcg {aid} {rec['cat']} maps {sorted(rec['files'])}", flush=True)
    man["acg"] = {
        ax: {
            f"acg_{aid}": {
                "kind": "tex",
                "id": aid,
                "cat": cat,
                "src": "ambientcg",
                "tint_class": SR.TINT_CLASS.get(cat, "natural"),
            }
            for aid, cat in v
        }
        for ax, v in pick.items()
    }


def gso(r: Path, man: dict) -> None:
    models = []
    for page in range(1, 30):
        try:
            got = get_json(SR.GSO_API.format(page=page))
        except urllib.error.HTTPError as e:  # 404 after the last page
            if e.code != 404:
                raise
            got = []
        if not got:
            break
        models += got
    names = SR.pick_gso(models)
    meta = {m["name"]: m for m in models}

    def one(name):
        dims = gso_fetch(r, name)
        rec = {
            "src": "gso",
            "license": SR.SOURCES["gso"]["license"],
            "attribution": SR.SOURCES["gso"]["attribution"],
            "usd": f"models/gso/{name}/model.usda",
            "dims_mm": [round(x * 1000, 1) for x in dims],
            "page": f"https://app.gazebosim.org/GoogleResearch/fuel/models/{name}",
            "categories": meta[name].get("categories") or [],
            "description": meta[name].get("description", ""),
        }
        why = AC.size_why(dims)
        if why:
            rec["excluded"] = why
        return name, rec

    with capacity.sized(STEP, len(names)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        for name, rec in ex.map(one, names):
            man["models"][f"gso:{name}"] = rec
            print(f"{MARK} gso {name} {rec['dims_mm']} {rec.get('excluded', '')}", flush=True)
    man["gso_pick"] = names


def canonical_list() -> tuple[dict, dict[str, str]]:
    """The list of the canonical generation kept in git (LIST_DIR): (manifest, {file: sha256}). It must be that
    generation (sources.ASSET_GEN: sha256 of its SHA256SUMS · rows · bytes); a missing or other list dies."""
    for p in (LIST_DIR / "SHA256SUMS", LIST_DIR / "manifest.json"):
        if not p.is_file():
            raise InputMissingError(f"{MARK} canonical list {p} missing: it is the fetch input (no live choice)")
    raw = (LIST_DIR / "SHA256SUMS").read_bytes()
    sums = {f: h for h, f in (ln.split("  ", 1) for ln in raw.decode("utf-8").splitlines() if ln.strip())}
    man = json.loads((LIST_DIR / "manifest.json").read_text(encoding="utf-8"))
    got = {
        "sums_sha256": hashlib.sha256(raw).hexdigest(),
        "files": len(sums),
        "bytes": man.get("bytes"),
        "manifest_sha256": hashlib.sha256((LIST_DIR / "manifest.json").read_bytes()).hexdigest(),
    }
    want = {k: SR.ASSET_GEN[k] for k in got}
    if got != want:
        raise SystemExit(f"{MARK} {LIST_DIR} is not the canonical {SR.ASSET_GEN['name']} list: {got} != {want}")
    return man, sums


def check(r: Path, sums: dict[str, str]) -> None:
    """Every row of the list is a file of r with that sha256 and r holds no other asset file; else die naming them."""
    have = {
        str(p.relative_to(r))
        for p in r.rglob("*")
        if p.is_file() and p.name not in NOT_ASSETS and not p.name.endswith(".part")
    }
    differ = [f for f in sorted(sums) if f in have and hashlib.sha256((r / f).read_bytes()).hexdigest() != sums[f]]
    missing, extra = sorted(set(sums) - have), sorted(have - set(sums))
    if differ or missing or extra:
        raise SystemExit(
            f"{MARK} {r} is not the canonical {SR.ASSET_GEN['name']} (list {LIST_DIR / 'SHA256SUMS'}): sha256 differs "
            f"{len(differ)} {differ[:5]} · missing {len(missing)} {missing[:5]} · not listed {len(extra)} {extra[:5]} "
            "(a file changed upstream or left by another fetch: remove it and fetch again)"
        )


def write_license(r: Path, man: dict) -> None:
    lic = ["Third-party assets in this folder (per-asset source · license · page in manifest.json):\n"]
    for k, v in SR.SOURCES.items():
        n = sum(1 for tab in (man["textures"], man["models"], man["hdris"]) for x in tab.values() if x.get("src") == k)
        lic.append(
            f"- {v['name']} ({v['url']}) — {v['license']} ({v['license_url']}) — {n} assets"
            + (f"\n  Attribution: {v['attribution']}" if v["attribution"] else "")
            + "\n"
        )
    (r / "LICENSE.txt").write_text("".join(lic))


def materialise() -> dict:
    """Fetch the canonical generation into the root: ids from its list, every file checked against its row, then the
    list's SHA256SUMS and manifest.json installed as they are."""
    r = root()
    man, sums = canonical_list()
    mf = r / "manifest.json"
    if mf.is_file() and (got := hashlib.sha256(mf.read_bytes()).hexdigest()) != SR.ASSET_GEN["manifest_sha256"]:
        was = " (the canonical root's original manifest)" if got == SR.ASSET_GEN["manifest_from"] else ""
        raise SystemExit(
            f"{MARK} {r} already holds a manifest.json of sha256 {got}{was}, not the list's "
            f"{SR.ASSET_GEN['manifest_sha256']}: the fetch does not rewrite another root's manifest; "
            "fetch into a new root"
        )
    ph = sorted(
        ("texture" if tab == "textures" else "model", pid, rec["res"])
        for tab in ("textures", "models")
        for pid, rec in man[tab].items()
        if rec["src"] == "polyhaven"
    )
    if ph != AC.ph_ids():  # the fixed tables of the code and the list must name the same Poly Haven assets
        raise SystemExit(
            f"{MARK} augvis.catalog.ph_ids() differs from the canonical list {LIST_DIR}: {ph} != {AC.ph_ids()}"
        )
    jobs = [(f"polyhaven {k} {pid} {res}", partial(ph_fetch, r, k, pid, res)) for k, pid, res in ph]
    jobs += [(f"hdri {pid}", partial(hdri_fetch, r, pid)) for pid in sorted(man["hdris"])]
    acg = sorted({e["id"] for ents in man["acg"].values() for e in ents.values()})
    jobs += [(f"ambientcg {aid}", partial(acg_fetch, r, aid)) for aid in acg]
    jobs += [(f"gso {name}", partial(gso_fetch, r, name)) for name in man["gso_pick"]]
    t0 = time.time()
    with capacity.sized(STEP, len(jobs)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        for (what, _fn), _got in zip(jobs, ex.map(lambda j: j[1](), jobs), strict=True):
            print(f"{MARK} {what}", flush=True)
    check(r, sums)
    for f in ("SHA256SUMS", "manifest.json"):
        (r / f).write_bytes((LIST_DIR / f).read_bytes())
    write_license(r, man)
    print(f"{MARK} DONE {len(jobs)} assets · files {len(sums)}/{len(sums)} · {time.time() - t0:.0f} s", flush=True)
    return man


def build_new() -> dict:
    """`--new-generation`: today's live choice with a rebuilt manifest and list (not the canonical generation)."""
    r = root()
    man = {
        "sources": SR.SOURCES,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "textures": {},
        "models": {},
        "hdris": {},
        "acg": {},
        "gso_pick": [],
        "look_d": look_d_albedo(),
    }
    for nm, fn in (("polyhaven", polyhaven), ("hdri", hdris), ("ambientcg", ambientcg), ("gso", gso)):
        t0 = time.time()
        fn(r, man)
        print(f"{MARK} {nm} done · {time.time() - t0:.0f} s", flush=True)
    sums = {}
    for p in sorted(r.rglob("*")):
        if p.is_file() and p.name not in NOT_ASSETS and not p.name.endswith(".part"):
            sums[str(p.relative_to(r))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (r / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in sums.items()))
    man["files"], man["bytes"] = len(sums), sum((r / f).stat().st_size for f in sums)
    (r / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    write_license(r, man)
    n_ok = len(man["textures"]) + len(man["models"]) + len(man["hdris"])
    print(
        f"{MARK} DONE new generation {n_ok} · textures {len(man['textures'])} · models {len(man['models'])} · "
        f"HDRI {len(man['hdris'])} · files {man['files']} · {man['bytes'] / 2**20:.1f} MiB · {r} — not the canonical "
        f"{SR.ASSET_GEN['name']}: commit its list at configs/record/<name>/ and name it in sources.ASSET_GEN",
        flush=True,
    )
    return man


def verify() -> int:
    """The root is the canonical generation: every listed file with its sha256, nothing else, the list's manifest."""
    r = root()
    _man, sums = canonical_list()
    check(r, sums)
    got = hashlib.sha256((r / "manifest.json").read_bytes()).hexdigest() if (r / "manifest.json").is_file() else None
    if got != SR.ASSET_GEN["manifest_sha256"]:  # the statement of the list's (translated) manifest
        raise SystemExit(
            f"{MARK} {r}/manifest.json sha256 {got} is not the canonical list's {SR.ASSET_GEN['manifest_sha256']} "
            f"(a root fetched before the translation holds {SR.ASSET_GEN['manifest_from'][:12]}…: fetch again)"
        )
    print(f"{MARK} check passed · files {len(sums)} · {SR.asset_gen_line(r)}", flush=True)
    return len(sums)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only check the root against the canonical list")
    ap.add_argument(
        "--new-generation",
        action="store_true",
        help="today's live choice and a new list (not the canonical generation: the render dies on it)",
    )
    config.add_data_root_arg(ap)
    a = ap.parse_args()
    config.apply_data_root(a)
    if a.new_generation:
        build_new()
    else:
        if not a.verify:
            materialise()
        verify()
