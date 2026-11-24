#!/usr/bin/env python3
"""ManipTrans residual rows: the authors' object convex hulls, computed once (a speed and stability change; the hulls
are identical). Isaac Gym computes V-HACD for every collision mesh of an object URDF at load (the authors' AssetOptions,
maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py:743-760) and caches each result in <getpwuid home>/.isaacgym/vhacd/
<mesh path with / and . -> _>.vhacd. A container without that path recomputes it in every row. Here, on CPU PhysX with
the same options, every object of the 64 residual rows is loaded once into a fresh cache; the cache files the rows need
are packed under data/maniptrans/hull_cache/ (vhacd/ · manifest.json {sequence: [cache file]} · SHA256SUMS). A job
binds them at its <getpwuid home>/.isaacgym; a miss computes exactly as the authors do.
usage (the ManipTrans env python, this repository on PYTHONPATH): hull_cache.py build | prove <object id> …
(prove = sha256 of a fresh decomposition twice, byte identity; run before build ships anything)"""

import hashlib
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments import workspace

CACHE = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".isaacgym/vhacd"  # where Isaac Gym writes (getpwuid, not $HOME)


def obj_dir() -> Path:
    return workspace.data("maniptrans/objects")


def out_dir() -> Path:
    return workspace.data("maniptrans/hull_cache")


LOADER = r"""
import os, sys
from isaacgym import gymapi
gym = gymapi.acquire_gym()
for urdf in sys.argv[1:]:
    sp = gymapi.SimParams()
    sp.use_gpu_pipeline = False
    sp.physx.use_gpu = False
    sim = gym.create_sim(-1, -1, gymapi.SIM_PHYSX, sp)
    o = gymapi.AssetOptions()
    o.override_com = True
    o.override_inertia = True
    o.convex_decomposition_from_submeshes = True
    o.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    o.thickness = 0.001
    o.max_linear_velocity = 50
    o.max_angular_velocity = 100
    o.fix_base_link = False
    o.vhacd_enabled = True
    o.vhacd_params = gymapi.VhacdParams()
    o.vhacd_params.resolution = 200000
    o.density = 200
    a = gym.load_asset(sim, *os.path.split(urdf), o)
    print("[HULL_LOAD]", urdf, gym.get_asset_rigid_shape_count(a), flush=True)
    gym.destroy_sim(sim)
"""


def cache_name(mesh: Path) -> str:
    return re.sub(r"[/.]", "_", str(mesh).lstrip("/")) + ".vhacd"


def meshes(oid: str) -> list:
    u = obj_dir() / oid / f"{oid}_maniptrans.urdf"
    return [u.parent / m.get("filename") for c in ET.parse(u).getroot().iter("collision") for m in c.iter("mesh")]


def rows() -> list:
    """The dataIndices of the 64 residual rows (run/maniptrans_cmds.txt, maniptrans/gen_cmds.sh)."""
    q = workspace.root() / "run/maniptrans_cmds.txt"
    s = [
        re.search(r"dataIndices=\[([^\]]+)\]", ln).group(1)
        for ln in q.read_text().splitlines()
        if ln and not ln.startswith("#")
    ]
    assert len(s) == 64 and len(set(s)) == 64, len(s)
    return s


def objects(seq: str) -> list:
    import pyarrow.parquet as pq

    p = (
        pq.read_table(workspace.data("chord_hf_all_taco") / f"{seq}.parquet", columns=["object_mesh_paths"])
        .column(0)[0]
        .as_py()
    )
    return sorted({os.path.basename(m).split("_")[0] for m in p})


def load(oids: list) -> None:
    urdfs = [str(obj_dir() / o / f"{o}_maniptrans.urdf") for o in oids]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", LOADER, *urdfs], env=env, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0 or r.stdout.count("[HULL_LOAD]") != len(urdfs):
        raise SystemExit(f"[HULL_CACHE] FAIL loader rc={r.returncode}: {r.stderr[-400:]}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def drop(oids: list) -> None:
    for o in oids:
        for m in meshes(o):
            (CACHE / cache_name(m)).unlink(missing_ok=True)


def prove(oids: list) -> None:
    drop(oids)
    load(oids)
    a = {cache_name(m): sha(CACHE / cache_name(m)) for o in oids for m in meshes(o)}
    drop(oids)
    load(oids)
    b = {cache_name(m): sha(CACHE / cache_name(m)) for o in oids for m in meshes(o)}
    for o in oids:
        ns = [cache_name(m) for m in meshes(o)]
        same = all(a[n] == b[n] for n in ns)
        print(
            f"[HULL_PROVE] object {o}: {len(ns)} meshes · fresh A vs fresh B sha256 "
            f"{'IDENTICAL' if same else 'DIFFERENT'} · "
            f"first {ns[0][-24:]} {a[ns[0]][:12]} / {b[ns[0]][:12]}"
        )
    if a != b:
        raise SystemExit("[HULL_PROVE] FAIL not byte-identical — nothing is shipped")
    OUT = out_dir()
    if (OUT / "vhacd").is_dir():  # the packed cache (build) vs this fresh decomposition, file by file
        sh = {n: sha(OUT / "vhacd" / n) for n in b if (OUT / "vhacd" / n).is_file()}
        for o in oids:
            ns = [cache_name(m) for m in meshes(o)]
            ok = all(sh.get(n) == b[n] for n in ns)
            print(
                f"[HULL_PROVE] object {o}: packed cache vs fresh decomposition sha256 "
                f"{'IDENTICAL' if ok else 'DIFFERENT'} "
                f"({sum(n in sh for n in ns)}/{len(ns)} files packed)"
            )
            if not ok:
                raise SystemExit("[HULL_PROVE] FAIL packed cache differs from a fresh decomposition")


def build() -> None:
    OUT = out_dir()
    seqs = rows()
    per = {s: objects(s) for s in seqs}
    oids = sorted({o for v in per.values() for o in v})
    drop(oids)
    load(oids)  # a fresh cache for every object the rows load
    tmp = OUT.with_name(OUT.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "vhacd").mkdir(parents=True)
    man = {}
    for s in seqs:
        ns = [cache_name(m) for o in per[s] for m in meshes(o)]
        for n in ns:
            if not (CACHE / n).is_file():
                raise SystemExit(f"[HULL_CACHE] FAIL {s}: no cache file {n} after the load")
            shutil.copy2(CACHE / n, tmp / "vhacd" / n)
        man[s] = ns
    (tmp / "manifest.json").write_text(json.dumps(man, indent=0, sort_keys=True) + "\n")
    sums = "".join(f"{sha(f)}  vhacd/{f.name}\n" for f in sorted((tmp / "vhacd").iterdir()))
    (tmp / "SHA256SUMS").write_text(sums)
    if OUT.exists():
        shutil.rmtree(OUT)
    tmp.rename(OUT)
    print(
        f"[HULL_CACHE] built {len(seqs)} rows · {len(oids)} objects · "
        f"{len(list((OUT / 'vhacd').iterdir()))} cache files · "
        f"sha {hashlib.sha256(sums.encode()).hexdigest()[:12]} → {OUT}"
    )


if __name__ == "__main__":
    if sys.argv[1] == "prove":
        prove(sys.argv[2:])
    elif sys.argv[1] == "build":
        build()
    else:
        raise SystemExit(__doc__)
