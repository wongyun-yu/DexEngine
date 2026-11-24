"""Look D assets: the look name and log marker, the MDL materials and office props it places, the scene desk (Willow),
and their local mirror (build, verify, resolve; look_d.py applies the look and settles the lighting).

The Willow desk of every scene (dexengine.rl.tasks.scene.desk_visual) lives only in this mirror: it is NVIDIA content
(Commercial Assets Pack, the Isaac Sim Additional Software and Materials License forbids redistribution —
THIRD_PARTY.md), so no tree carries a copy. `python -m dexengine.record.looks.look_d --mirror` downloads it from
NVIDIA's S3 (each user downloads it under NVIDIA's terms) and checks every file
against WILLOW_SHA256; willow_usd() refuses a missing or changed file."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from dexengine.core.errors import InputMissingError

LOOK = "D"  # render look of the recording path


MARK = "[COLLECT_LOOK]"


S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"


MIRROR_REL = "data/assets/scene_canon/collect_look_d"


_ISAAC = "Assets/Isaac/4.5"


MDL = {
    "ceiling": f"{_ISAAC}/NVIDIA/Materials/Base/Architecture/Ceiling_Tiles.mdl",
    "floor": f"{_ISAAC}/NVIDIA/Materials/Base/Masonry/Concrete_Polished.mdl",
    "wall": f"{_ISAAC}/NVIDIA/Materials/Base/Wall_Board/Plaster.mdl",
    "desk": f"{_ISAAC}/NVIDIA/Materials/Base/Wood/Birch.mdl",
}


_OFFICE = f"{_ISAAC}/Isaac/Environments/Office/Props"


# (asset, wall 'nx'|'px'|'ny'|'py' or floor '-', coordinate along the wall ((x, y) on the floor), yaw deg); props
# stand with their back to the wall, on the floor
PROPS = (
    (f"{_OFFICE}/SM_Rack.usd", "nx", -1.4, 0.0),
    (f"{_OFFICE}/SM_FileCabinet_01.usd", "nx", 0.9, 0.0),
    (f"{_OFFICE}/SM_Plant01.usd", "nx", 2.2, 0.0),
    (f"{_OFFICE}/SM_Cupboard.usd", "px", 0.3, 0.0),
    (f"{_OFFICE}/SM_ChairOffice.usd", "-", (1.45, 1.0), 200.0),
    (f"{_OFFICE}/SM_BoxBig.usd", "px", -1.6, 0.0),
)


#: The Willow desk on NVIDIA's S3 (mirror-relative) and the sha256 of each file — the bytes every scene was built with
#: (the copy the repository once carried, measured equal to S3).
WILLOW = "Assets/ArchVis/Commercial/Tables/Willow.usd"
WILLOW_SHA256 = {
    "Assets/ArchVis/Commercial/Tables/Willow.usd": "72b5163f640827abdc436ec340aee7a414e75a046718bbf3d0795cd5c9320793",
    "Assets/ArchVis/Commercial/Tables/Textures/WillowTable_Metal_A.png": (
        "c3c31b9ad1f85d0a13e37fb073e2e51d8b30b5ca61081ffb3dd7843ed455c373"
    ),
    "Assets/ArchVis/Commercial/Tables/Textures/WillowTable_Metal_R.png": (
        "337bf441c97b77eb72bd36a710fd582c0952a0d1a4fde0c79934d689a920fdfe"
    ),
    "Assets/ArchVis/Commercial/Tables/Textures/WillowTable_Pad_A.png": (
        "337bf441c97b77eb72bd36a710fd582c0952a0d1a4fde0c79934d689a920fdfe"
    ),
    "Assets/ArchVis/Commercial/Tables/Textures/WillowTable_Pad_R.png": (
        "6d1d35281fbc94c6ad83d4c88ecadecc4b6c444e7ba823cb9070fe3034b227f3"
    ),
    "Assets/ArchVis/Commercial/Tables/Textures/WillowTable_Wood_A.png": (
        "216304cdc218489f3687b602bc54c80e234bcee5f06f6548bef25068d41e70cf"
    ),
    "Assets/ArchVis/Commercial/Tables/Textures/WillowTable_Wood_N.png": (
        "dde145fd8acdcd806a726214bbaaee2d457fdcfc72dadb45ea6f8a33e2c35d86"
    ),
    "Assets/ArchVis/Commercial/Tables/Textures/WillowTable_Wood_R.png": (
        "74a880652497a3f2c37e2d2d767ac49ccb7bd0dcd430a43bc9f384a1bd67081e"
    ),
}
WILLOW_NOTICE = (
    "NVIDIA content (Commercial Assets Pack, Isaac Sim Additional Software and Materials License): not redistributed — "
    "each user downloads it from NVIDIA under NVIDIA's terms"
)


# ---- local mirror ----
def mirror_root() -> Path:
    from dexengine.core import paths

    return paths.main_root() / MIRROR_REL


def _base() -> list[str]:
    from dexengine.record.looks import materials as LB

    return LB.base_roots()


def local(rel: str) -> str:
    p = mirror_root() / rel
    if not p.is_file():
        raise InputMissingError(
            f"{MARK} {rel} missing from the local mirror ({p}) — run "
            "`python -m dexengine.record.looks.look_d --mirror` (no S3 fallback)"
        )
    return str(p)


def _download(rel: str, root: Path) -> Path:
    import urllib.request

    dst = root / rel
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(f"{S3}/{rel}", timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(dst)
    return dst


def _deps(local_path: Path, rel: str) -> list[str]:
    """Direct dependencies of one file (mirror-relative): USD via pxr (unresolved paths of
    UsdUtils.ComputeAllDependencies), MDL via relative texture_2d("...") paths and same-folder modules."""
    base = os.path.dirname(rel)
    out = []
    if local_path.suffix.lower() == ".mdl":
        src = local_path.read_text(errors="replace")
        out.extend(
            os.path.normpath(os.path.join(base, m.group(1)))
            for m in re.finditer(r'texture_2d\(\s*"([^"]+)"', src)
            if not m.group(1).startswith("/")
        )
        # same-folder module (.::OmniUe4Base)
        out.extend(
            os.path.normpath(os.path.join(base, m.group(1) + ".mdl"))
            for m in re.finditer(r"(?:using|import)\s+\.::(\w+)", src)
        )
        return out
    if local_path.suffix.lower() in (".usd", ".usda", ".usdc"):
        from pxr import Sdf, UsdUtils

        layer = Sdf.Layer.FindOrOpen(str(local_path))
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(layer.identifier)
        here = os.path.dirname(str(local_path))
        for u in unresolved:
            u = str(u)
            r = os.path.relpath(u, str(mirror_root())) if os.path.isabs(u) else os.path.normpath(os.path.join(here, u))
            r = os.path.relpath(r, str(mirror_root())) if os.path.isabs(r) else r
            out.append(os.path.normpath(r))
        for a in list(_assets) + [x.realPath for x in _layers if x.realPath]:  # already downloaded dependencies
            a = str(a)
            if a.startswith(str(mirror_root())):
                out.append(os.path.relpath(a, str(mirror_root())))
    return out


def base_files() -> list[str]:
    """Mirror-relative files the base render look (materials.BASE_PRESET, a training's render) reads: its MDLs and
    what they open (_deps over the local mirror; Kit built-in Omni* modules excluded as in build_mirror). A missing
    file dies."""
    root, queue, seen = mirror_root(), _base(), set()
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        if rel.lower().endswith(".mdl") and Path(rel).name.startswith("Omni") and not (root / rel).is_file():
            continue
        if not (root / rel).is_file():
            raise InputMissingError(f"{MARK} base look file missing in the mirror: {root / rel}")
        seen.add(rel)
        queue += [d for d in _deps(root / rel, rel) if d not in seen and not d.startswith("..")]
    return sorted(seen)


def willow_files() -> list[Path]:
    """The Willow desk's files in the mirror (WILLOW_SHA256 order) — what every scene reads for its desk."""
    return [mirror_root() / rel for rel in WILLOW_SHA256]


def willow_usd(check_sha: bool = True) -> str:
    """The Willow desk USD in the local mirror. A missing file or one whose sha256 differs from WILLOW_SHA256 raises
    ConfigError naming the download step (no other desk, no S3 at spawn time)."""
    from dexengine.core.config import ConfigError

    key = (str(mirror_root()), check_sha)
    if key in _WILLOW_OK:
        return str(mirror_root() / WILLOW)
    bad = [
        str(p)
        for p, h in zip(willow_files(), WILLOW_SHA256.values(), strict=True)
        if not p.is_file() or (check_sha and hashlib.sha256(p.read_bytes()).hexdigest() != h)
    ]
    if bad:
        raise ConfigError(
            f"{MARK} Willow desk missing or changed in the local mirror: {bad[:3]}{' …' if len(bad) > 3 else ''} — "
            f"run `python -m dexengine.record.looks.look_d --mirror` ({WILLOW_NOTICE})"
        )
    _WILLOW_OK.add(key)
    return str(mirror_root() / WILLOW)


_WILLOW_OK: set = set()  # (mirror root, check_sha) already checked in this process


def build_mirror() -> dict:
    root = mirror_root()
    print(f"{MARK} Willow desk from NVIDIA's S3 — {WILLOW_NOTICE}", flush=True)
    roots = list(MDL.values()) + [p[0] for p in PROPS] + _base() + list(WILLOW_SHA256)
    queue, seen = list(roots), set()
    import urllib.error

    builtin = []
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            lp = _download(rel, root)
        except urllib.error.HTTPError as e:
            # only Kit built-in MDL modules (Omni... names, resolved by the MDL search path) are skipped
            if e.code in (403, 404) and rel.lower().endswith(".mdl") and Path(rel).name.startswith("Omni"):
                builtin.append(rel)
                seen.discard(rel)
                continue
            raise SystemExit(f"{MARK} mirror download failed {rel}: {e}") from e
        queue += [d for d in _deps(lp, rel) if d not in seen and not d.startswith("..")]
    files = sorted(seen)
    sums = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
    if bad := sorted(f for f, h in WILLOW_SHA256.items() if sums.get(f) != h):
        raise SystemExit(f"{MARK} Willow desk differs from WILLOW_SHA256 after download: {bad}")
    (root / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in sums.items()))
    man = {
        "look": LOOK,
        "source": S3,
        "files": len(files),
        "bytes": sum((root / f).stat().st_size for f in files),
        "roots": roots,
        "builtin_mdl_skipped": sorted(set(builtin)),
    }
    (root / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{MARK} mirror {root} · files {man['files']} · {man['bytes'] / 2**20:.1f} MiB", flush=True)
    return man


def verify_mirror(check_sha: bool = True, with_base: bool = True) -> int:
    """Check the mirror list (and sha256). with_base=False when applying look D: the base-layer files are checked by
    rl.render.scene_look at binding."""
    root = mirror_root()
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise InputMissingError(f"{MARK} no mirror list {sums} — run --mirror first")
    bad = []
    rows = [ln.split("  ", 1) for ln in sums.read_text().splitlines() if ln.strip()]
    for h, f in rows:
        p = root / f
        if not p.is_file() or (check_sha and hashlib.sha256(p.read_bytes()).hexdigest() != h):
            bad.append(f)
    listed = {f for _h, f in rows}
    bad.extend(
        f"(not listed) {rel}"
        for rel in list(MDL.values()) + [p[0] for p in PROPS] + (_base() if with_base else []) + list(WILLOW_SHA256)
        if rel not in listed
    )
    if bad:
        raise SystemExit(f"{MARK} mirror mismatch {len(bad)} {bad[:5]}")
    return len(rows)
