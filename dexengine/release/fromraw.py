"""From-raw release gate: an empty root that holds only the official downloads and the shipped inputs, then the
public pipeline from the start.

    python -m dexengine.release.fromraw prepare <ds> --root <new dir> --downloads <dir> --mano <dir> --shipped <dir>
        [--depth-standin <dir>]
    python -m dexengine.release.fromraw run <ds> --root <prepared root> [--workers N] [--rgbd <dir>]

``prepare`` makes ``<root>`` a ``fromraw_<ds>`` generation root (dexengine.core.layout.generations) holding only:
  * per-file links to the dataset's official downloads — the globs of READERS[<ds>].downloads, relative to the raw
    layer, found under ``--downloads`` (the download folder laid out like the raw layer); nothing else of the raw
    layer is linked, so every file the pipeline derives from the downloads comes from this run;
  * per-file links to the MANO model files from ``--mano`` (``data/body_models/mano``, a download of its own);
  * copies of the shipped inputs of this dataset (configs/release/shipped_inputs.tsv rows whose path is ``data`` or
    lies under ``data/<ds>/`` or ``data/<ds>_…/``), each checked against its digest ([SHIPPED] state=verified, or it
    dies).
It writes ``FROMRAW.json`` (dataset, code commit, the run environment, the links and copies) and prints
``[FROMRAW_ROOT] ds=… links=… copies=… other=0 standin=…``. A root that already exists dies.

H2O's desk plane reads cam4 depth frames 0 and 100 from the RGB-D release (770 GB, ``readers.h2o_unpack depth``;
one pass over the four archives takes about 20–30 h at the measured 7–11 MB/s).
``--depth-standin <dir>`` links an already extracted ``sample`` folder instead; the record then says ``standin`` and
the gate cannot pass on such a root.

``run`` (the ``mano`` and ``taco`` drivers; the others die as not yet run by the gate). TACO: stage 1 for
``--arm/--hand`` over every clip of the shipped source (scripts/stage1_retarget.py), then the INTAKE derivations;
``--subset <list>`` runs the sample clips and marks the record. ``mano``: the reader with
READERS[<ds>].read_args writes the whole MANO layer (``--workers`` processes over the clip list it prints), the
object assets are built from the raw meshes (dexengine.data.objects.build --mano), H2O streams the desk-plane depth
frames out of the RGB-D archives in ``--rgbd`` unless the root holds a stand-in, and the dataset's INTAKE derivations
run (``read_raw <ds> derive``). Each step's log is kept under ``<root>/_fromraw_logs``; its rc and end marker are
checked and its time is added to FROMRAW.json; ``[FROMRAW_RUN] ds=… DONE`` ends it.

Gate scope: the MANO layer and INTAKE are rebuilt for the whole dataset (values fitted over the whole
dataset — the hand direction, the support summaries — cannot be reproduced from a sample); per-hand stage 1 onward
runs on K sample clips (TACO: ``--subset``). Pass rule: from the raw downloads, every file ``outputs`` lists on
either of two machines with different CPUs (one with AVX-512, one without; stage 1 on the pinned OpenBLAS kernel,
dexengine.core.blas_pin) gives the same verdict view — the verdicts (adopted · rejected · reason), labels, counts and
structure; float values are reported as max |Δ| with no threshold and stamps are left out by tree_content's
definition (dexengine.release.compare --write-ref on one, --ref on the other; compare_rules.tsv). Data made by older
code is not the reference.

    python -m dexengine.release.fromraw outputs --root <root>     (the files a gate compares, one path per line)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dexengine.core import capacity, shipped
from dexengine.core.errors import InputMissingError
from dexengine.core.layout import generations as GEN
from dexengine.core.layout.layers import render
from dexengine.core.repo import REPO_ROOT
from dexengine.release.compare import run_environment

TAG = "[FROMRAW_ROOT]"
RUN_TAG = "[FROMRAW_RUN]"
RECORD = "FROMRAW.json"
READ_STEP = "dexengine.release.fromraw:read"  # capacity.NEEDS key: one reader process per part of the clip list
MANO_REL = "data/body_models/mano"
MANO_FILES = ("MANO_RIGHT.pkl", "MANO_LEFT.pkl")
STANDIN_REL = "sample"  # h2o: the folder readers.h2o_unpack depth writes (h2o_unpack.DEPTH_DIR)


def _link_tree(src_root: Path, rel_glob: str, dst_root: Path) -> int:
    n = 0
    for f in sorted(src_root.glob(rel_glob)):
        if f.is_file():
            d = dst_root / f.relative_to(src_root)
            d.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(f.resolve(), d)
            n += 1
    return n


def shipped_rows(ds: str) -> list[str]:
    """Shipped inputs of `ds`: rows whose path is `data` (every dataset) or lies under data/<ds>/ or data/<ds>_…/."""
    out = []
    for name, r in shipped.rows().items():
        p = r["path"]
        top = p.split("/")[1] if p.count("/") >= 1 else ""
        if p == "data" or top == ds or top.startswith(ds + "_"):
            out.append(name)
    return out


def _check_inputs(ds: str, globs: tuple[str, ...], downloads: Path, mano: Path, shipped_dir: Path) -> None:
    """Every stated input is there before the root is made (a missing one dies and leaves no partial root)."""
    for g in globs:
        if not any(f.is_file() for f in downloads.glob(g)):
            raise InputMissingError(f"{TAG} no file of the download {g} under {downloads}")
    for f in MANO_FILES:
        if not (mano / f).is_file():
            raise InputMissingError(f"{TAG} {mano / f} missing (MANO model download)")
    for name in shipped_rows(ds):
        if not (shipped_dir / name).is_dir():
            raise InputMissingError(f"{TAG} shipped input {name} missing: {shipped_dir / name}")


def prepare(ds: str, root: Path, downloads: Path, mano: Path, shipped_dir: Path, standin: Path | None) -> dict:
    from dexengine.data import readers as R  # noqa: PLC0415

    globs = R.spec(ds).downloads
    if not globs:
        raise InputMissingError(f"{TAG} READERS[{ds}].downloads states no official download — the gate cannot run {ds}")
    if root.exists():
        raise SystemExit(f"{TAG} ✗ {root} exists — the gate builds a new, empty root")
    if standin is not None and ds != "h2o":
        raise SystemExit(f"{TAG} ✗ --depth-standin is the H2O desk-plane frames only")
    _check_inputs(ds, globs, downloads, mano, shipped_dir)
    raw_rel = render("raw", ds)
    root.mkdir(parents=True)
    (root / GEN.GENERATION_FILE).write_text(json.dumps(GEN.generation_record(GEN.FROMRAW_PREFIX + ds), sort_keys=True))
    links = {g: _link_tree(downloads, g, root / raw_rel) for g in globs}
    (root / MANO_REL).mkdir(parents=True, exist_ok=True)
    for f in MANO_FILES:
        os.symlink((mano / f).resolve(), root / MANO_REL / f)
    if standin is not None:
        links[f"{STANDIN_REL} (standin)"] = _link_tree(standin, "**/*", root / raw_rel / STANDIN_REL)
    copies = {}
    for name in shipped_rows(ds):
        src, dst = shipped_dir / name, root / shipped.row(name)["path"]
        shutil.copytree(src, dst, dirs_exist_ok=True)
        if shipped.state(name, dst) != shipped.VERIFIED:
            raise SystemExit(f"{TAG} ✗ shipped input {name} is not verified at {dst}")
        copies[name] = len(shipped.members(name, dst))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True).stdout
    rec = {
        "dataset": ds,
        "code": {"tree": str(REPO_ROOT), "commit": commit.strip()},
        "environment": run_environment(),
        "downloads": str(downloads),
        "links": links,
        "copies": copies,
        "standin": standin is not None,
    }
    (root / RECORD).write_text(json.dumps(rec, indent=1, sort_keys=True))
    n_links = sum(1 for p in root.rglob("*") if p.is_symlink())
    want_links = sum(links.values()) + len(MANO_FILES)
    if n_links != want_links:
        raise SystemExit(f"{TAG} ✗ {n_links} links in {root}, stated {want_links}")
    print(
        f"{TAG} ds={ds} links={n_links} copies={sum(copies.values())} ({', '.join(copies)}) other=0 "
        f"standin={int(standin is not None)} → {root}",
        flush=True,
    )
    return rec


def _step(root: Path, name: str, argv: list[str], must: str, rec: dict) -> str:
    """Run one step in the gate root (data root = the gate root, code = this tree); returns its log text."""
    logs = root / "_fromraw_logs"
    logs.mkdir(exist_ok=True)
    env = {  # the gate root is the data root; this tree and this interpreter run every step and every child
        **os.environ,
        "DEXENGINE_MAIN_ROOT": str(root),
        "PYTHONPATH": str(REPO_ROOT),
        "RETARGET_PY": sys.executable,
        "OMP_NUM_THREADS": "1",
    }
    t0 = time.time()
    with open(logs / f"{name}.log", "w") as fh:
        r = subprocess.run([sys.executable, *argv], cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    text = (logs / f"{name}.log").read_text(errors="replace")
    rec.setdefault("run", {})[name] = {"rc": r.returncode, "s": round(time.time() - t0, 1)}
    if r.returncode or (must and must not in text):
        last = [x for x in text.splitlines() if x.strip()][-1:] or [""]
        raise SystemExit(f"{RUN_TAG} ✗ {name} rc={r.returncode} (marker {must!r}) {last[0][:160]}")
    print(f"{RUN_TAG} {name} rc=0 {rec['run'][name]['s']} s", flush=True)
    return text


def _run_taco(root: Path, arm: str, hand: str, subset: Path | None, rec: dict) -> int:
    """TACO: stage 1 (lift … ik for `hand`) over every clip of the shipped source, or the sample clips of `subset`
    (recorded as `subset`), then the INTAKE derivations."""
    from dexengine.core.layout import taco_source_root  # noqa: PLC0415
    from dexengine.data.readers import taco_source  # noqa: PLC0415

    allc = taco_source.clips(root / taco_source_root())
    clips = allc if subset is None else [c for c in subset.read_text().splitlines() if c.strip()]
    if not set(clips) <= set(allc):
        raise SystemExit(f"{RUN_TAG} ✗ {len(set(clips) - set(allc))} listed clips are not in the TACO source")
    bases = root / "_fromraw_logs" / "bases.txt"
    bases.parent.mkdir(exist_ok=True)
    bases.write_text("".join(c + "\n" for c in clips))
    rec["subset"] = subset is not None
    argv = ["scripts/stage1_retarget.py", "--bases", str(bases), "--arm", arm, "--hand", hand, "--dataset", "taco"]
    _step(
        root,
        f"stage1_{hand}",
        [*argv, "--emb", "bimanual", "--log-dir", str(root / "_fromraw_logs" / "stage1")],
        "STATUS=DONE",
        rec,
    )
    return len(clips)


def run(ds: str, root: Path, workers: int, rgbd: Path | None, **taco) -> dict:
    from dexengine.data import readers as R  # noqa: PLC0415

    rec = json.loads((root / RECORD).read_text())
    spec = R.spec(ds)
    if spec.driver == "taco":
        n = _run_taco(root, taco["arm"], taco["hand"], taco.get("subset"), rec)
        _step(root, "derive", ["-m", "dexengine.data.read_raw", ds, "derive"], "derive DONE", rec)
        rec["clips"] = n
        (root / RECORD).write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{RUN_TAG} ds={ds} clips={n} subset={int(rec['subset'])} DONE", flush=True)
        return rec
    if spec.driver != "mano":
        raise SystemExit(f"{RUN_TAG} ✗ the {spec.driver} driver is not yet run by the gate ({ds})")
    raw, mano = root / render("raw", ds), root / render("mano", ds, emb="bimanual")
    base = ["-m", spec.module, "--raw", str(raw), "--mano-dir", str(root / MANO_REL), *spec.read_args]
    clips = [c for c in _step(root, "list", [*base, "--list"], "", rec).splitlines() if c.strip()]
    if not clips:
        raise SystemExit(f"{RUN_TAG} ✗ the reader lists no clip under {raw}")
    with capacity.sized(READ_STEP, len(clips), workers) as cap, ThreadPoolExecutor(cap.workers) as ex:
        parts = [p for p in (clips[i :: cap.workers] for i in range(cap.workers)) if p]
        list(
            ex.map(
                lambda k: _step(root, f"read_{k}", [*base, "--out", str(mano), "--clips", *parts[k]], "", rec),
                range(len(parts)),
            )
        )
    got = sum(1 for c in clips if (mano / c / "0/trajectory_keypoints.npz").is_file())
    if got != len(clips):
        raise SystemExit(f"{RUN_TAG} ✗ MANO layer {got}/{len(clips)} clips")
    objs = [
        "-m",
        "dexengine.data.objects.build",
        ds,
        "--raw",
        str(raw),
        "--out",
        str(root / render("assets_objects", ds)),
    ]
    _step(root, "objects", [*objs, "--mano", str(mano), "--workers", str(workers)], "[OBJ_BUILD_DONE]", rec)
    if ds == "h2o" and not rec["standin"]:
        archives = sorted(rgbd.glob("subject*_v1_1.tar.gz")) if rgbd else []
        if not archives:
            raise InputMissingError(f"{RUN_TAG} h2o needs --rgbd <folder of subject<k>_v1_1.tar.gz> (desk-plane depth)")
        for a in archives:
            _step(
                root,
                f"depth_{a.name}",
                ["-m", "dexengine.data.readers.h2o_unpack", "depth", "--archive", str(a)],
                "[H2O_UNPACK] depth",
                rec,
            )
    _step(root, "derive", ["-m", "dexengine.data.read_raw", ds, "derive"], "derive DONE", rec)
    rec["clips"] = len(clips)
    (root / RECORD).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"{RUN_TAG} ds={ds} clips={len(clips)} DONE", flush=True)
    return rec


def outputs(root: Path) -> list[str]:
    """Root-relative paths of every regular file under <root>/data that is not a link: what the run made, plus the
    shipped copies. The official downloads and MANO files are links, so they are left out."""
    data = root / "data"
    if not data.is_dir():
        raise SystemExit(f"{TAG} ✗ {root} has no data folder")
    return sorted(str(p.relative_to(root)) for p in data.rglob("*") if p.is_file() and not p.is_symlink())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dexengine.release.fromraw", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("dataset")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--downloads", type=Path, required=True)
    p.add_argument("--mano", type=Path, required=True, help="folder with MANO_RIGHT.pkl and MANO_LEFT.pkl")
    p.add_argument("--shipped", type=Path, required=True, help="folder with one subfolder per shipped input")
    p.add_argument("--depth-standin", type=Path, default=None)
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--rgbd", type=Path, default=None, help="h2o: folder of the RGB-D archives subject<k>_v1_1.tar.gz")
    r.add_argument("--arm", default=None, help="taco: the arm of stage 1 (no default)")
    r.add_argument("--hand", default=None, help="taco: the hand of stage 1 (no default)")
    r.add_argument("--subset", type=Path, default=None, help="taco: the sample clips (one per line)")
    o = sub.add_parser("outputs", help="list the files a gate compares (one root-relative path per line)")
    o.add_argument("--root", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "outputs":
        print("\n".join(outputs(a.root)))
        return 0
    if a.cmd == "prepare":
        prepare(a.dataset, a.root, a.downloads, a.mano, a.shipped, a.depth_standin)
    else:
        if a.dataset == "taco" and not (a.arm and a.hand):
            ap.error("run taco needs --arm and --hand")
        run(a.dataset, a.root, a.workers, a.rgbd, arm=a.arm, hand=a.hand, subset=a.subset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
