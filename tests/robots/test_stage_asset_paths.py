"""Every robot asset a stage opens resolves, for every hand mounted on rby1 — through each stage's own resolver:
f0 (repair_cfg.HandCfg.urdf, both sides) · cross repair and judge (sim_collider.combo_urdf / resolved_urdf) ·
RL spawn (selection.combo_usd_path) · recording (the plate, selection._usd_literal with plate_variant) ·
RL scene build of a sampled tree with an articulated main object (the bundle input statement; its link model is
measured from the articulated reader and must be named by the statement's float-settle rows — the oracle of
tests/placement/test_run_inputs_artic.py). The usd_artic generation layer is baked into a scratch overlay by its CPU
writer (surface points · keypoints: artic.surface); the articulated USD and its cooked dump need Kit (PhysX schema ·
cooking), so here the stated USD path is compared with the path the RL reader opens — USD existence is measured by the
from-raw gate (Kit stage) and the RL preflight.
A URDF also has every mesh it names resolved by the stages' rule (paths.urdf_mesh). Meshes and USDs that live under
the data root (data/…) are checked when the data root holds them (DEXENGINE_MAIN_ROOT, the canonical machine);
anything in the repository is always checked."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.core.mounts import MOUNT_REGISTRY

ARM = "rby1"
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DEXENGINE_MAIN_ROOT", "/nonexistent"))
HAS_DATA = (DATA_ROOT / "data" / "_stage").is_dir()  # vendor files placed (`python -m dexengine.robots.vendor fetch`)
#: built robot USDs and canonical trees are separate layers: a root with only the fetched vendor files has neither
HAS_USD = (DATA_ROOT / "data" / "usd_cache").is_dir()
HANDS = sorted(h for (a, h), m in MOUNT_REGISTRY.items() if a == ARM and m is not None)
#: canonical trees whose scene build opens object files beyond the object USD: (dataset, emb, side, clip) — an
#: articulated main object (its link model: the tree scene.xml and link meshes); the second one has a welded body
SCENE_SAMPLE = [
    ("oakink2", "bimanual", "both", "oakink2_insert_usb__scene_02_788108ba88cd_s001_pl_rt"),
    ("oakink2", "bimanual", "both", "oakink2_close_gate__scene_01_9f436c64de60_s008_pl_rt"),
]
HAS_TREES = all((DATA_ROOT / "data" / ds / "trees").is_dir() for ds, *_ in SCENE_SAMPLE)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """The real data root (captured at import — the suite's autouse fixture points each test at an empty one)."""
    monkeypatch.setenv("DEXENGINE_ARM", ARM)
    if HAS_DATA:
        monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(DATA_ROOT))


def _missing_meshes(urdf: Path) -> tuple[list[str], int]:
    """(missing mesh refs, refs skipped because they live under an absent data root)."""
    miss, skipped = [], 0
    for n in sorted(set(re.findall(r'filename="([^"]+)"', urdf.read_text(encoding="utf-8")))):
        p = Path(P.urdf_mesh(urdf, n))
        if p.is_file():
            continue
        if "data/" in n and not HAS_DATA:
            skipped += 1
            continue
        miss.append(n)
    return miss, skipped


def _urdf_resolves(urdf) -> None:
    urdf = Path(urdf)
    assert urdf.is_file(), f"URDF missing: {urdf}"
    miss, _skipped = _missing_meshes(urdf)
    assert not miss, f"{urdf.name}: {len(miss)} meshes do not resolve, e.g. {miss[:3]}"


def test_hands_are_the_canonical_four():
    assert len(HANDS) == 4, HANDS


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize("side", ["right", "left"])
def test_f0_urdf(hand, side, monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", hand)
    from dexengine.retarget.f0 import repair_cfg as RC

    _urdf_resolves(RC.HandCfg(hand, ARM).urdf(side))


@pytest.mark.parametrize("hand", HANDS)
def test_cross_repair_and_judge_urdf(hand, monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", hand)
    monkeypatch.delenv("DEXENGINE_XR_URDF", raising=False)
    from dexengine.placement.collision import sim_collider as SC

    _urdf_resolves(SC.combo_urdf(ARM, hand))
    if hasattr(SC, "resolved_urdf"):
        _urdf_resolves(SC.resolved_urdf(ARM, hand))


@pytest.mark.skipif(not HAS_USD, reason="robot USDs live under the data root (DEXENGINE_MAIN_ROOT)")
@pytest.mark.parametrize("hand", HANDS)
def test_rl_and_record_usd(hand, monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", hand)
    from dexengine.core import arms as AR
    from dexengine.robots import selection as SEL

    usd_dir = str(DATA_ROOT / "data" / "usd_cache")
    assert Path(SEL.combo_usd_path(ARM, hand, usd_dir)).is_file()
    assert Path(SEL._usd_literal(ARM, hand, usd_dir, AR.plate_variant(ARM, hand))).is_file()


def test_a_missing_mesh_is_caught(tmp_path):
    """Negative: a URDF naming a mesh that is not there fails the check."""
    u = tmp_path / "x.urdf"
    u.write_text(
        '<robot name="x"><link name="a"><visual><geometry><mesh filename="nope.stl"/></geometry></visual>'
        "</link></robot>",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="do not resolve"):
        _urdf_resolves(u)


def _scene_probe(ds: str, emb: str, side: str, clip: str, overlay: str, ids: str) -> None:
    """Run in a fresh process on the real data root: bake the usd_artic files of articulated assets `ids` with their
    CPU writer into `overlay` (same relative paths), and run the RL reader's --sequence_dir resolution for `clip`
    (its preflight errors recorded, not raised). Prints one JSON line: written (overlay-relative) · reader_usd."""
    import argparse
    import hashlib
    import json

    from dexengine.core import artifact_paths as TP
    from dexengine.data.artic import surface as SF
    from dexengine.rl.cli_entry import seqdir as SD

    root, ov, tpd = Path(os.environ["DEXENGINE_MAIN_ROOT"]), Path(overlay), TP.get(ds)
    specs, sha = SF._load_specs(ds), hashlib.sha256(SF.SPEC_FILE.read_bytes()).hexdigest()[:12]
    written = []
    for oid in filter(None, ids.split(",")):
        spec, mesh_dir = specs[oid], SF._assets_root(ds) / specs[oid]["obj_id"]
        r = SF.build_one(oid, spec, mesh_dir, SF._theta_of(SF.THETA_DEFAULT, spec))
        out_dir, sp, kp = (ov / Path(f).relative_to(root) for f in SF.asset_files(tpd, oid))
        SF._write_surface(r, out_dir, sp, kp, mesh_dir, sha)
        written += [str(f.relative_to(ov)) for f in (sp, kp)]

    class _Parser:
        def error(self, msg):
            print(f"[probe] reader preflight: {msg[:200]}", file=sys.stderr)

    class _Args(argparse.Namespace):
        def __getattr__(self, k):
            return None

    a = _Args(arm=ARM, hand="sharpa_wave", sequence=None, bimanual=int(side == "both"), dual_eval=0, multi_object=0)
    a.left_uni, a.dataset = int(side == "left"), ds
    a.sequence_dir = str(root / "data" / ds / "trees" / f"{ARM}_sharpa_wave" / emb)
    SD._resolve_sequence_dir(_Parser(), a)
    usd = {u for k in ("object_usd_right", "object_usd_left") for u in (getattr(a, k) or "").split(",") if u}
    print(json.dumps({"written": sorted(written), "reader_usd": sorted(usd)}))


@pytest.fixture(scope="module", params=SCENE_SAMPLE, ids=lambda t: t[3])
def scene(request, tmp_path_factory):
    """(statement items, measured link model, articulated asset ids, probe result, overlay) of a sampled tree."""
    if not HAS_TREES:
        pytest.skip("trees and object files live under the data root (DEXENGINE_MAIN_ROOT)")
    import json

    ds, emb, side, clip = request.param
    mp = pytest.MonkeyPatch()
    mp.setenv("DEXENGINE_MAIN_ROOT", str(DATA_ROOT))
    mp.setenv("DEXENGINE_ARM", ARM)
    from dexengine.core import artifact_paths as TP
    from dexengine.placement.bundle import run_inputs as RI
    from tests.placement.test_run_inputs_artic import artic_reads

    combo = f"{ARM}_sharpa_wave"
    items = RI.clip_inputs(clip, combo=combo, hand="sharpa_wave", dataset=ds, emb=emb, side=side)
    td = DATA_ROOT / "data" / ds / "trees" / combo / emb / clip
    measured = artic_reads(td, json.loads((td / "task_info.json").read_text(encoding="utf-8")))
    art = Path(TP.get(ds).artic_usd_root())
    mp.undo()
    ids = sorted({_abs(it.path).relative_to(art).parts[0] for it in items if _abs(it.path).is_relative_to(art)})
    tmp = tmp_path_factory.mktemp("overlay")
    (tmp / "seq.txt").write_text(clip + "\n", encoding="utf-8")
    env = dict(os.environ, DEXENGINE_MAIN_ROOT=str(DATA_ROOT), DEXENGINE_ARM=ARM, DEXENGINE_HAND="sharpa_wave")
    env |= {"DEXENGINE_DATASET": ds, "DEXENGINE_EMBODIMENT": emb, "DEXENGINE_SEQ_LIST": str(tmp / "seq.txt")}
    code = f"from tests.robots.test_stage_asset_paths import _scene_probe as f; f(*{[ds, emb, side, clip]!r}, "
    code += f"{str(tmp / 'ov')!r}, {','.join(ids)!r})"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    probe = json.loads(out.stdout.strip().splitlines()[-1])
    out = {"items": items, "measured": measured, "ids": ids, "probe": probe, "overlay": tmp / "ov", "art": art}
    return out | {"ds": ds, "tree": td}


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else DATA_ROOT / p


def _check_artic_layer(stated: set, usd: set, written: set) -> None:
    """The statement's usd_artic files of the sample = the Kit-made USDs + the files the CPU writer wrote."""
    assert written, "the usd_artic overlay was not baked"
    assert stated - usd == written, (sorted(map(str, stated - usd - written)), sorted(map(str, written - stated)))


def _check_usd_paths(stated_usd: set, reader_usd: set) -> None:
    """The object USDs the statement names for spawning = the ones the RL reader opens (one owner: artifact_paths)."""
    assert stated_usd == reader_usd, (
        sorted(map(str, stated_usd - reader_usd)),
        sorted(map(str, reader_usd - stated_usd)),
    )


def _parts(sc) -> tuple[set, set, set, set]:
    """(stated usd_artic files, their Kit-made USDs, written files re-rooted on the data root, stated spawn USDs)."""
    from dexengine.core import artifact_paths as TP

    stated = {_abs(it.path) for it in sc["items"]}
    art_files = {p for p in stated if p.is_relative_to(sc["art"])}
    usd = {Path(TP.get(sc["ds"]).usd_file(i, articulated=True)) for i in sc["ids"]}
    written = {DATA_ROOT / w for w in sc["probe"]["written"]}
    spawn = {_abs(it.path) for it in sc["items"] if it.reader == "entry:preflight" and str(it.path).endswith(".usd")}
    return art_files, usd, written, spawn


def test_scene_build_files_resolve(scene):
    """The sample holds an articulated main object; every file its reader opens is named by the statement's
    float-settle rows (the measurement oracle); its usd_artic files are the Kit-made USD plus what the CPU writer
    bakes into the overlay; every other stated file exists on the data root; the stated spawn USDs are the RL
    reader's (USD existence: the from-raw gate)."""
    from tests.placement.test_run_inputs_artic import unstated

    stated = {_abs(it.path) for it in scene["items"]}
    assert len(scene["measured"]) > 1, "the sample must hold an articulated main object"
    assert not unstated(scene["measured"], scene["items"]), sorted(
        map(str, unstated(scene["measured"], scene["items"]))
    )
    art_files, usd, written, spawn = _parts(scene)
    _check_artic_layer(art_files, usd, written)
    assert all((scene["overlay"] / w).is_file() for w in scene["probe"]["written"])
    miss = sorted(str(p) for p in stated - art_files if not p.exists())
    assert not miss, f"{len(miss)} stated files do not resolve on the data root, e.g. {miss[:3]}"
    _check_usd_paths(spawn, {Path(u) for u in scene["probe"]["reader_usd"]})


def test_the_gate_fails_without_the_overlay(scene):
    """Negative: with nothing baked, the usd_artic check fails."""
    art_files, usd, _written, _spawn = _parts(scene)
    with pytest.raises(AssertionError, match="not baked"):
        _check_artic_layer(art_files, usd, set())


def test_the_gate_fails_on_a_planted_usd_path(scene):
    """Negative: a spawn USD path the RL reader does not open (a planted other path) fails the comparison."""
    _art, _usd, _written, spawn = _parts(scene)
    planted = {p.with_name("planted_" + p.name) if p.is_relative_to(scene["art"]) else p for p in spawn}
    with pytest.raises(AssertionError):
        _check_usd_paths(planted, {Path(u) for u in scene["probe"]["reader_usd"]})


def test_the_gate_fails_on_a_link_dropped_from_the_owner(scene, monkeypatch):
    """Negative: float_settle.inputs without one link mesh the reader opens fails the oracle on that file; restored,
    it passes."""
    from dexengine.core import artifact_paths as TP
    from dexengine.data.objects import float_settle as FS
    from dexengine.placement.bundle import run_inputs as RI
    from tests.placement.test_run_inputs_artic import unstated

    gone = max(p for p in scene["measured"] if p.name != "scene.xml")
    owner, tpd = FS.inputs, TP.get(scene["ds"])
    monkeypatch.setattr(FS, "inputs", lambda tree_dir, t: [f for f in owner(tree_dir, t) if Path(f).resolve() != gone])
    assert unstated(scene["measured"], RI.scene_inputs(scene["tree"], tpd)) == {gone}
    monkeypatch.setattr(FS, "inputs", owner)
    assert not unstated(scene["measured"], RI.scene_inputs(scene["tree"], tpd))
