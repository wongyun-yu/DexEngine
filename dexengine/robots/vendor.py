"""Robot and hand source files: fetched from the manufacturers' public repositories or copied from this repository.

WHY
    The repository ships no file a manufacturer publishes. ``configs/robots/vendor/sources.yaml`` names each source
    (repository, pinned commit, licence) and ``files.tsv`` lists every file with its destination and sha256. A
    manufacturer file is downloaded from the pinned commit; a file the manufacturer does not publish is shipped under
    the source's ``root`` and copied. Either way the bytes must equal the listed sha256, or nothing is written.

USAGE
    python -m dexengine.robots.vendor fetch --data-root DIR [--only rby1,dg5f_s]
        download / copy, check, place (skips good files), then the scene look mirror (--only look_mirror)
    python -m dexengine.robots.vendor check --data-root DIR   # every file present with its sha256 (rc 1 if not)
    python -m dexengine.robots.vendor build --data-root DIR   # the derived assets, in order (docs/hands.md)

    A destination that starts with ``data/`` is placed under ``--data-root`` (``paths.repo_rel_path``), any other one
    under this checkout. A root that already holds generated data needs ``--allow-data-root`` (check reads only).
    Marker ``[VENDOR]``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

import yaml

from dexengine.core import paths as P

MARK = "[VENDOR]"
VENDOR_DIR = P.CONFIGS_DIR / "robots" / "vendor"
UA = {"User-Agent": "dexengine-vendor-fetch/1.0"}
KINDS = ("github", "repository")


class Row(NamedTuple):
    source: str
    dest: str
    path: str
    sha256: str


def sources(vendor_dir: Path = VENDOR_DIR) -> dict:
    src = yaml.safe_load((vendor_dir / "sources.yaml").read_text(encoding="utf-8"))
    for name, s in src.items():
        if s.get("kind") not in KINDS:
            raise SystemExit(f"{MARK} source {name}: kind {s.get('kind')!r} — one of {KINDS}")
        need = ("repo", "commit") if s["kind"] == "github" else ("root",)
        missing = [k for k in need if not s.get(k)]
        if missing and not (s["kind"] == "github" and set(missing) == set(need)):
            raise SystemExit(f"{MARK} source {name}: missing {missing}")
    return src


def unset_sources(src: dict) -> set[str]:
    """Sources this repository leaves for the user to fill: kind github with an empty `repo` and `commit` (and an empty
    sha256 column in files.tsv). Fetching one dies; the hands it serves are not available until it is filled."""
    return {n for n, s in src.items() if s["kind"] == "github" and not s.get("repo") and not s.get("commit")}


UNSET = f"{MARK} source not set:"


def _unset_message(names: set[str], vendor_dir: Path = VENDOR_DIR) -> str:
    n = sum(1 for r in rows(vendor_dir) if r.source in names)
    where = vendor_dir.relative_to(P.REPO_ROOT) if vendor_dir.is_relative_to(P.REPO_ROOT) else vendor_dir
    return (
        f"{UNSET} {sorted(names)} — {where}/sources.yaml leaves its `repo` "
        f"and `commit` empty and files.tsv its sha256 column ({n} files) for you to fill"
    )


def rows(vendor_dir: Path = VENDOR_DIR) -> list[Row]:
    out = []
    for i, ln in enumerate((vendor_dir / "files.tsv").read_text(encoding="utf-8").splitlines(), 1):
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) != 4 or len(parts[3]) not in (0, 64):
            raise SystemExit(f"{MARK} files.tsv line {i}: want source, destination, path, sha256 — {ln!r}")
        out.append(Row(*parts))
    dup = {r.dest for r in out if sum(1 for s in out if s.dest == r.dest) > 1}
    if dup:
        raise SystemExit(f"{MARK} files.tsv: destinations listed twice {sorted(dup)[:5]}")
    return out


def dest_path(row: Row) -> Path:
    return P.repo_rel_path(row.dest)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _download(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: PERF203 — transient network errors only, 3 times
            if attempt == 2:
                raise SystemExit(f"{MARK} download failed {url}: {e}") from e
            time.sleep(3)
    raise AssertionError


#: Git LFS pointer (https://git-lfs.github.com/spec/v1): a repository that stores a file in LFS serves this text at the
#: raw URL. Rule: a pointer whose oid is the listed sha256 is followed to the LFS object (GitHub's media URL at the same
#: commit); a pointer with another oid is a different file and fails the sha256 check like any other bytes.
LFS_HEAD = b"version https://git-lfs.github.com/spec/v1\n"


def lfs_oid(b: bytes) -> str | None:
    """The sha256 oid of a Git LFS pointer, or None when the bytes are not a pointer."""
    if not b.startswith(LFS_HEAD) or len(b) > 1024:
        return None
    for ln in b.decode("ascii", "replace").splitlines():
        if ln.startswith("oid sha256:"):
            return ln.split(":", 1)[1].strip()
    return None


def source_bytes(src: dict, row: Row) -> bytes:
    if src["kind"] == "github":
        b = _download(f"https://raw.githubusercontent.com/{src['repo']}/{src['commit']}/{row.path}")
        if row.sha256 and lfs_oid(b) == row.sha256:
            b = _download(f"https://media.githubusercontent.com/media/{src['repo']}/{src['commit']}/{row.path}")
        return b
    return (P.REPO_ROOT / src["root"] / row.path).read_bytes()


def _good(p: Path, sha: str) -> bool:
    return p.is_file() and _sha(p.read_bytes()) == sha


def fetch(only: set[str] | None = None, vendor_dir: Path = VENDOR_DIR) -> dict[str, int]:
    """Place every listed file (of ``only`` sources). A file already there with its sha256 is left; a mismatch after
    download raises before anything is written for that file."""
    src, todo = sources(vendor_dir), rows(vendor_dir)
    unknown = sorted({r.source for r in todo} - set(src)) + sorted((only or set()) - set(src))
    if unknown:
        raise SystemExit(f"{MARK} unknown source(s) {unknown} — sources.yaml names {sorted(src)}")
    unset = unset_sources(src) & (only or set(src))
    if unset:
        others = ",".join(sorted(set(src) - unset_sources(src)))
        hint = f"or fetch the other sources with --only {others},{LOOK_MIRROR}"
        raise SystemExit(f"{_unset_message(unset, vendor_dir)}; {hint}")
    blank = sorted({r.source for r in todo if not r.sha256 and (not only or r.source in only)})
    if blank:
        raise SystemExit(f"{MARK} source(s) {blank}: files.tsv rows without a sha256 — fill the column")
    n = {"placed": 0, "present": 0}
    for r in todo:
        if only and r.source not in only:
            continue
        d = dest_path(r)
        if _good(d, r.sha256):
            n["present"] += 1
            continue
        b = source_bytes(src[r.source], r)
        if _sha(b) != r.sha256:
            raise SystemExit(f"{MARK} {r.source}:{r.path} sha256 {_sha(b)[:12]} ≠ listed {r.sha256[:12]} — not written")
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(d.name + f".part{os.getpid()}")
        tmp.write_bytes(b)
        tmp.replace(d)
        n["placed"] += 1
    print(f"{MARK} fetch placed={n['placed']} present={n['present']} sources={sorted(only or src)}", flush=True)
    return n


#: The scene look's local mirror (NVIDIA materials, props and the Willow desk; not redistributable) is the other
#: outside download. Its owner is dexengine.record.looks.look_assets (file list, WILLOW_SHA256); `fetch` only calls it,
#: the same as `python -m dexengine.record.looks.look_d --mirror`.
LOOK_MIRROR = "look_mirror"


def fetch_look_mirror() -> int:
    """Build the look mirror when it is missing or its files differ (a Willow desk whose sha256 differs from
    WILLOW_SHA256 stops the build), then verify it. Returns the number of mirror files."""
    from dexengine.core.errors import InputMissingError
    from dexengine.record.looks import look_assets as LA

    try:
        n = LA.verify_mirror()
        print(f"{MARK} look mirror present ({n} files, sha256 checked) {LA.mirror_root()}", flush=True)
        return n
    except (SystemExit, InputMissingError) as e:
        print(f"{MARK} look mirror incomplete ({e}) — building it", flush=True)
    LA.build_mirror()
    return LA.verify_mirror()


def check(vendor_dir: Path = VENDOR_DIR, only: set[str] | None = None) -> list[str]:
    """Problems (missing or other bytes, a source not set) over every listed file (of ``only`` sources); empty = all
    present."""
    unset = unset_sources(sources(vendor_dir))
    probs = [_unset_message({n}, vendor_dir) for n in sorted(unset) if only is None or n in only]
    for r in rows(vendor_dir):
        if r.source in unset or (only is not None and r.source not in only):
            continue
        d = dest_path(r)
        if not d.is_file():
            probs.append(f"missing {r.dest}")
        elif _sha(d.read_bytes()) != r.sha256:
            probs.append(f"sha256 differs {r.dest}")
    return probs


def require(*dests: str) -> None:
    """Die with the fetch command when a listed vendor file is not in place (callers that read vendor files)."""
    listed = {r.dest: r for r in rows()}
    for d in dests:
        if d in listed and not dest_path(listed[d]).is_file():
            raise SystemExit(f"{MARK} {d} is not in place — run `python -m dexengine.robots.vendor fetch`")


def _body_mounts() -> list[tuple[str, str]]:
    """Every registered (arm, hand) mount of an arm with a body (``MOUNT_REGISTRY``; a floating hand has no merged
    URDF)."""
    from dexengine.core import arms as AS
    from dexengine.core.mounts import MOUNT_REGISTRY

    return sorted(
        (a, h) for (a, h), m in MOUNT_REGISTRY.items() if m is not None and not AS.ARM_REGISTRY[a].is_floating
    )


def hand_sources(vendor_dir: Path = VENDOR_DIR) -> dict[str, set[str]]:
    """{hand: the sources it owns}: the source named after the hand and the sources of its vendor documents (HandSpec
    vendor_urdf / vendor_mjcf). A source no hand owns (the arm, the head camera) is shared."""
    from dexengine.core import hands

    src, by_dest = sources(vendor_dir), {r.dest: r.source for r in rows(vendor_dir)}
    out = {}
    for name, hs in hands.HAND_REGISTRY.items():
        if not hs:
            continue
        docs = {d.format(side=s) for d in (hs.vendor_urdf, hs.vendor_mjcf) if d for s in ("right", "left")}
        out[name] = ({name} & set(src)) | {by_dest[d] for d in docs if d in by_dest}
    return out


def required_sources(hands: set[str], vendor_dir: Path = VENDOR_DIR) -> set[str]:
    """The sources a build of ``hands`` reads: the shared ones and those the hands own (``hand_sources``)."""
    own = hand_sources(vendor_dir)
    unknown = sorted(set(hands) - set(own))
    if unknown:
        raise SystemExit(f"{MARK} unknown hand(s) {unknown} — registered {sorted(own)}")
    owned = set().union(*own.values())
    return (set(sources(vendor_dir)) - owned) | set().union(*(own[h] for h in hands))


def unavailable_hands(vendor_dir: Path = VENDOR_DIR) -> dict[str, str]:
    """{hand: reason} for the hands whose vendor documents (HandSpec vendor_urdf / vendor_mjcf) come from a source that
    is not set (``unset_sources``): nothing of theirs can be fetched or built until the source is filled."""
    from dexengine.core import hands

    unset = unset_sources(sources(vendor_dir))
    by_dest = {r.dest: r.source for r in rows(vendor_dir)}
    out = {}
    for name, hs in hands.HAND_REGISTRY.items():
        if not hs:
            continue
        docs = {d.format(side=s) for d in (hs.vendor_urdf, hs.vendor_mjcf) if d for s in ("right", "left")}
        miss = sorted({by_dest.get(d) for d in docs} & unset)
        if miss:
            out[name] = f"source {miss} not set (configs/robots/vendor/sources.yaml)"
    return out


def build_steps(hands: set[str] | None = None) -> list[list[str]]:
    """The derived assets, in build order (each step reads the previous one's output), for every body mount
    (``_body_mounts``) of ``hands`` (None = every hand): the hand rigs, the merged robot URDFs, then the joint-limit
    copies of the hands whose sim body is a limit variant (``arms.sim_hand_limits``). The USD bake is the next step
    (scripts/build_robot_usd.py, Isaac Lab python) and is not run here."""
    from dexengine.core import arms as AS

    pairs = [(a, h) for a, h in _body_mounts() if hands is None or h in hands]
    hands = sorted({h for _, h in pairs})
    limited = sorted({h for a, h in pairs if AS.sim_hand_limits(a, h)})
    steps = [["dexengine/robots/hand_rig.py", h] for h in hands]
    steps += [["dexengine/robots/urdf/merge.py", f"{a}+{h}"] for a, h in pairs]
    steps += [["dexengine/robots/urdf/truncate.py", "--hand", h] for h in limited]
    return steps


def built_outputs(hands: set[str] | None = None) -> list[Path]:
    """The checkout files `build` writes (one per build step: the right-hand rig, the merged URDF, its limit copy)."""
    from dexengine.core import arms
    from dexengine.core import hands as HS
    from dexengine.core.repo import FULL_ROBOT_URDF_DIR

    out = []
    for step in build_steps(hands):
        if step[0].endswith("hand_rig.py"):
            out.append(P.hand_rig_dir(HS.HAND_REGISTRY[step[1]].retarget_dirname or step[1]) / "right.xml")
        elif step[0].endswith("merge.py"):
            d = P.combo_dirname(step[1].replace("+", "_"))
            out.append(FULL_ROBOT_URDF_DIR / d / f"{d}.urdf")
        else:  # the limit copy (dexengine/robots/urdf/truncate.py: <canonical dir><TRUNCATED_HAND_LIMITS>)
            d = P.combo_dirname(P.combo_for(step[2])) + arms.TRUNCATED_HAND_LIMITS
            out.append(FULL_ROBOT_URDF_DIR / d / f"{d}.urdf")
    return out


def missing_in_checkout() -> list[str]:
    """Checkout files that `fetch` (destinations outside data/) or `build` would write for the available hands and
    that are not in place — what a fresh clone lacks. Empty = this checkout can run the robot code of those hands
    (``unavailable_hands`` names the others)."""
    avail = {h for _, h in _body_mounts()} - set(unavailable_hands())
    probs = [p for p in check() if " data/" not in p and not p.startswith(UNSET)]
    probs += [f"not built {p.relative_to(P.REPO_ROOT)}" for p in built_outputs(avail) if not p.is_file()]
    return probs


#: Layers only a data generation writes: a root that holds one of them is a data root (generated data lives there).
DATA_LAYER_MARKS = ("data/usd_cache", "data/*/trees", "data/*/retarget", "data/*/base")


def data_root_marks(root: Path) -> list[str]:
    """The generated-data layers present under ``root`` (empty = a fresh root)."""
    return sorted(str(m.relative_to(root)) for g in DATA_LAYER_MARKS for m in root.glob(g))[:3]


def target_root(given: str, allow_data_root: bool) -> Path:
    """The root `data/…` destinations go to, given explicitly. Pointing at a root that already holds generated data
    stops unless ``allow_data_root`` (a fetch overwrites files whose bytes differ). Sets the data root for this
    process (env DEXENGINE_MAIN_ROOT), so the look mirror goes there too."""
    root = Path(given).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{MARK} --data-root {root} is not a directory")
    marks = data_root_marks(root)
    if marks and not allow_data_root:
        raise SystemExit(
            f"{MARK} --data-root {root} holds generated data ({', '.join(marks)}) — writing vendor files there needs "
            "--allow-data-root"
        )
    os.environ[P.data_layout.DATA_ROOT_ENV] = str(root)
    print(f"{MARK} data root {root}{' (data root allowed)' if marks else ''}", flush=True)
    return root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("fetch", "check", "build"))
    ap.add_argument("--data-root", required=True, help="root that `data/…` destinations go to (no default)")
    ap.add_argument(
        "--allow-data-root",
        action="store_true",
        help="allow a --data-root that already holds generated data (usd_cache · trees · retarget · base)",
    )
    ap.add_argument("--only", default="", help="comma-separated source names of sources.yaml, or look_mirror (fetch)")
    ap.add_argument("--hands", default="", help="comma-separated hands to build (build; default every hand)")
    a = ap.parse_args()
    target_root(a.data_root, a.allow_data_root or a.action == "check")
    if a.action == "fetch":
        only = set(a.only.split(",")) if a.only else None
        if only is None or only - {LOOK_MIRROR}:
            fetch(only - {LOOK_MIRROR} if only else None)
        if only is None or LOOK_MIRROR in only:
            fetch_look_mirror()
        return 0
    if a.action == "check":
        probs = check()
        for p in probs[:20]:
            print(p if p.startswith(MARK) else f"{MARK} {p}")
        print(f"{MARK} check files={len(rows())} problems={len(probs)}")
        return 1 if probs else 0
    sel = set(a.hands.split(",")) if a.hands else {h for _, h in _body_mounts()}
    un = unavailable_hands()
    if sel & set(un):
        why = {h: un[h] for h in sorted(sel & set(un))}
        others = ",".join(sorted({h for _, h in _body_mounts()} - set(un)))
        raise SystemExit(
            f"{MARK} not available: {why} — fill the source, or build the other hands with --hands {others}"
        )
    need = required_sources(sel)
    print(f"{MARK} build hands {sorted(sel)} · reads sources {sorted(need)}", flush=True)
    probs = [p for p in check(only=need) if not p.startswith(UNSET)]
    if probs:
        raise SystemExit(f"{MARK} {len(probs)} vendor file(s) not in place ({probs[0]}) — run fetch first")
    py = P.require_retarget_py()
    for step in build_steps(sel):
        if step[0].endswith("hand_rig.py") and _rig_locked(step[1]):
            print(f"{MARK} build {step[1]} rig present with its registered hash — kept", flush=True)
            continue
        print(f"{MARK} build {' '.join(step)}", flush=True)
        subprocess.run([py, *step], cwd=P.REPO_ROOT, check=True)
    return 0


def _rig_locked(hand: str) -> bool:
    """The hand's rig folder already holds the rig its registry entry names (hand_rig.py refuses to overwrite it)."""
    from dexengine.core import hands

    name = hands.HAND_REGISTRY[hand].retarget_dirname or hand
    lock = P.data_layout.RETARGET_HAND_ASSETS.get(name)
    rig = P.hand_rig_dir(name)
    return lock is not None and (rig / "right.xml").is_file() and P.data_layout.asset_hash(str(rig)) == lock.sha256_xml3


if __name__ == "__main__":
    sys.exit(main())
