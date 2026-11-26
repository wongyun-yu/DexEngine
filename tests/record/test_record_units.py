"""Small unit checks of the record package (no Isaac)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_hooks_register_get_names():
    from dexengine.record import hooks

    obj = object()
    hooks.register("bake", "test_only", obj)
    assert hooks.get("bake", "test_only") is obj and "test_only" in hooks.names("bake")
    hooks.register("bake", "test_only", obj)  # same object again is fine
    with pytest.raises(ValueError):
        hooks.register("bake", "test_only", object())
    with pytest.raises(ValueError):
        hooks.register("nope", "x", obj)


def test_load_plugins_reads_env(tmp_path, monkeypatch):
    from dexengine.record import hooks

    (tmp_path / "rec_plugin_t.py").write_text(
        "from dexengine.record import hooks\nhooks.register('aug_session', 'rec_plugin_t', lambda: None)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert "rec_plugin_t" in hooks.load_plugins({hooks.PLUGINS_ENV: "rec_plugin_t"})
    assert hooks.get("aug_session", "rec_plugin_t")() is None


def test_isaac_cmd_runs_module():
    from dexengine.record.session.launch import isaac_cmd

    cmd = isaac_cmd("dexengine.record.session.entry", ["--stage", "render"])
    assert cmd[:2] == ["bash", "-c"] and "_isaac_env.sh" in cmd[2] and '-p -m "$0" "$@"' in cmd[2]
    assert cmd[3:] == ["dexengine.record.session.entry", "--stage", "render"]


def test_plugin_stage_unknown_is_none():
    from dexengine.record.session import launch

    assert launch.plugin_stage("record") is None and launch.plugin_stage("no_such_stage") is None
    assert launch.stage_timeout("render_clip") == 2400


def test_switches_metadata_values(monkeypatch):
    from dexengine.record import chain

    monkeypatch.setenv(chain.AUGC.ENV, "1")
    sw = chain.switches()
    flags = {"force_partner": True, "head_view": True, "aug_placement": True, "extras_overhang": True}
    assert sw == {**flags, "explicit": {chain.AUGC.ENV: "1"}}
    assert chain.render_tag() == "+head_view"


def test_episodes_per_session():
    from dexengine.record import episodes as EPI

    assert EPI.episodes_per_session(None) == EPI.EPISODES_PER_SESSION
    with pytest.raises(ValueError):
        EPI.episodes_per_session("share_look_session")


def test_read_table_threads_off(tmp_path):
    pa = pytest.importorskip("pyarrow")  # not installed in the Isaac interpreter
    pq = pytest.importorskip("pyarrow.parquet")

    from dexengine.record.store.parquet import read_table

    p = tmp_path / "t.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), p)
    assert read_table(p).column("a").to_pylist() == [1, 2, 3]
    with pytest.raises(TypeError):
        read_table(p, use_threads=True)


def test_scene_env_values():
    """The build-time scene inputs of a drawing dump (record.looks.scene_env): key light 3200 unless the caller set one,
    focal length always 12; no knob of the old render look is produced."""
    from dexengine.record.looks import scene_env as SE

    got = SE.env({})
    assert got == {"DEXENGINE_KEY_LIGHT": "3200", "DEXENGINE_CAMERA_FOCAL_MM": "12"}
    mine = SE.env({"DEXENGINE_KEY_LIGHT": "999", "DEXENGINE_CAMERA_FOCAL_MM": "8"})
    assert mine == {"DEXENGINE_CAMERA_FOCAL_MM": "12"}
    assert SE.env({"DEXENGINE_CAMERA_FOCAL_MM": "12", "DEXENGINE_KEY_LIGHT": "3200"}) == {}


def test_look_d_room_geometry():
    """Look D owns its room (no knob): four single-sided UV quads at /World/AugRoom (6 x 6 m, 2.6 m, centre (-0.7, 0)),
    read back as the bounds look_d.apply stores in render_meta."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from dexengine.record.looks import look_d as LOOK

    stage = Usd.Stage.CreateInMemory()
    LOOK._spawn_room(stage)
    room = LOOK._room(stage)
    assert room == pytest.approx({"x0": -3.7, "x1": 2.3, "y0": -3.0, "y1": 3.0, "h": 2.6})
    for nm in LOOK.WALLS:
        m = UsdGeom.Mesh(stage.GetPrimAtPath(f"{LOOK.ROOM_ROOT}/{nm}"))
        assert m.GetDoubleSidedAttr().Get() is False
        assert UsdGeom.PrimvarsAPI(m).GetPrimvar("st").IsDefined()


def test_stage4_record_dispatch_table():
    import runpy

    g = runpy.run_path(str(REPO / "scripts/stage4_record.py"), run_name="not_main")
    import importlib.util

    for mod in g["COMMANDS"].values():
        assert importlib.util.find_spec(mod) is not None, mod


def test_code_fingerprint_lists_exist():
    """Every file named in the stage-1/2 and augment code fingerprints exists (cache keys read them)."""
    from dexengine.record import cache
    from dexengine.record.augment import stage
    from dexengine.record.looks import look_d

    for f in (*cache.CODE_FILES, *cache.RENDER_CODE_FILES, *stage.AUG_CODE_FILES, *look_d.CODE_FILES):
        assert (REPO / f).is_file(), f
    assert len(cache.code_fingerprint(REPO)) > 0


def test_scene_layout_sidecar_roundtrip(tmp_path):
    """extract / write_sidecar on a synthetic final: fresh keys, provisional mark and statement file."""
    import json

    from dexengine.core import paths as P
    from dexengine.record.store import scene_layout as SL

    f = tmp_path / "collect" / "rby1_x"
    (f / "meta").mkdir(parents=True)
    eps = [
        {"episode_index": i, "clip": f"c{i}", "round": "orig", "side": "right", "body": {"root_z_offset_m": 0.01 * i}}
        for i in range(3)
    ]
    (f / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    x = SL.extract(f, "orig", False, None)
    assert (x["n"], x["have"], len(x["todo"]), x["prov_ok"]) == (3, 0, 3, True)
    line = lambda i, mark: {"episode_index": i, "clip": f"c{i}", "root_z_offset_m": 0.01 * i, "provisional": mark}  # noqa: E731
    assert SL.write_sidecar(f, [line(0, None), line(1, None)], None)["lines"] == 2
    x = SL.extract(f, "orig", False, None)
    assert (x["have"], len(x["todo"]), x["extra"]) == (2, 1, 0)
    prov = dict(hand="x", mark="P1", **{"from": "a"}, expiry="b", declared="c")
    assert SL.extract(f, "orig", False, prov)["prov_ok"] is False
    assert SL.write_sidecar(f, [line(2, "P1")], prov)["lines"] == 1  # unmarked old lines drop
    assert json.loads((f / "meta" / P.data_layout.PROVISIONAL_FILE).read_text()) == prov
    assert SL.write_sidecar(f, [], None)["lines"] == 0
    assert not (f / "meta" / P.data_layout.PROVISIONAL_FILE).exists()


def test_spec_mode_needs_site_plugin(monkeypatch, capsys):
    """Spec mode dies with rc 12 and names the plugin when no job-spec provider is registered."""
    from dexengine.record import driver_spec as DS
    from dexengine.record import hooks

    monkeypatch.setitem(hooks._REGISTRY, "driver_spec", {})
    monkeypatch.delenv(hooks.PLUGINS_ENV, raising=False)
    with pytest.raises(SystemExit) as e:
        DS._jobspec()
    assert e.value.code == DS.CODE["INPUT_MISSING"] == 12
    assert hooks.PLUGINS_ENV in capsys.readouterr().err


def test_owner_ledger_missing_plain_mode(tmp_path, capsys):
    """A missing owner ledger dies with rc 12 and says so, without needing the job-spec provider."""
    from dexengine.record import driver_spec as DS

    with pytest.raises(SystemExit) as e:
        DS.need_file(tmp_path / "rl_clip_owner.tsv")
    assert e.value.code == 12 and "owner ledger" in capsys.readouterr().err


@pytest.mark.parametrize("extra", ["redo_dir", "mirror_lane", "gate_cover", "regen_stale", "typo_key"])
def test_track_config_unknown_key_dies(tmp_path, capsys, extra):
    """track.json: every key must be known; an unknown one (the retired keys included) dies and names the key."""
    import json

    from dexengine.record.text import api as CTA
    from dexengine.record.text import track as TRK

    model = next(iter(CTA.PRICES))
    cfg = dict.fromkeys(TRK.KEYS)
    cfg.update(model=model, redo_model=model, **{extra: 1})
    (tmp_path / "track.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        TRK.load(tmp_path)
    assert f"unknown keys ['{extra}']" in capsys.readouterr().out


def test_track_config_known_keys_load(tmp_path):
    """A track.json with exactly the known keys loads (no state file = empty state)."""
    import json

    from dexengine.record.text import api as CTA
    from dexengine.record.text import track as TRK

    model = next(iter(CTA.PRICES))
    cfg = dict.fromkeys(TRK.KEYS)
    cfg.update(model=model, redo_model=model)
    (tmp_path / "track.json").write_text(json.dumps(cfg))
    got, st = TRK.load(tmp_path)
    assert got == cfg and st == {"rounds": [], "retexted": {}}


def _asset_root(tmp_path, files: dict) -> tuple:
    """A synthetic asset root: the files, their SHA256SUMS (augvis.assets row form) and the identity they make."""
    import hashlib

    for f, body in files.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes(body)
    rows = "".join(f"{hashlib.sha256(files[f]).hexdigest()}  {f}\n" for f in sorted(files))
    (tmp_path / "SHA256SUMS").write_text(rows)
    gen = {
        "name": "augvis_test",
        "sums_sha256": hashlib.sha256(rows.encode()).hexdigest(),
        "files": len(files),
        "bytes": sum(len(b) for b in files.values()),
    }
    return tmp_path, gen


def test_asset_generation_is_recognised_by_content(monkeypatch, tmp_path):
    """The canonical generation is its SHA256SUMS (sha256 · rows · listed bytes), not the fetch time; AUGVIS_ASSETS_ANY
    stays retired. A missing list, another list, a missing listed file or another size dies."""
    from dexengine.record.looks.augvis import sources as SR

    root, gen = _asset_root(tmp_path, {"textures/a.png": b"aa", "models/b/model.usd": b"bbbb"})
    monkeypatch.setattr(SR, "ASSET_GEN", gen)
    monkeypatch.setenv("AUGVIS_ASSETS_ANY", "1")
    (root / "manifest.json").write_text('{"created": "2030-01-01T00:00:00+0000"}')  # another fetch time passes
    assert SR.asset_gen_line(root) == f"[AUG_VIS] assets canonical augvis_test · files 2 · 6 B · {root}"
    (root / "models/b/model.usd").write_bytes(b"bbbbb")  # another size
    with pytest.raises(SystemExit, match="not the canonical"):
        SR.asset_gen_line(root)
    (root / "models/b/model.usd").unlink()  # a listed file missing
    with pytest.raises(SystemExit, match="listed files missing 1"):
        SR.asset_gen_line(root)
    (root / "models/b/model.usd").write_bytes(b"bbbb")
    assert SR.asset_gen_line(root).startswith("[AUG_VIS] assets canonical")
    sums = (root / "SHA256SUMS").read_text()
    (root / "SHA256SUMS").write_text(sums.replace("textures/a.png", "textures/c.png"))  # another list
    with pytest.raises(SystemExit, match="not the canonical"):
        SR.asset_gen_line(root)
    (root / "SHA256SUMS").unlink()
    with pytest.raises(SystemExit, match="SHA256SUMS missing"):
        SR.asset_gen_line(root)


def test_canonical_generation_keeps_its_counts():
    """The content key replaced only the fetch time: augvis_v2 is still 840 files and 864,779,498 B."""
    from dexengine.record.looks.augvis import sources as SR

    assert {k: SR.ASSET_GEN[k] for k in ("name", "files", "bytes")} == {
        "name": "augvis_v2",
        "files": 840,
        "bytes": 864779498,
    }
    assert len(SR.ASSET_GEN["sums_sha256"]) == 64


def test_persist_queue_under_tmpdir_and_env_retired(monkeypatch, tmp_path):
    """The worker queue lives under the job TMPDIR; the retired DEXENGINE_COLLECT_PERSIST_DIR is not read."""
    import tempfile

    from dexengine.record.session import persist as PS

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("DEXENGINE_COLLECT_PERSIST_DIR", str(tmp_path / "shared"))
    q = PS.queue_dir(tmp_path, "0")
    assert q.parent == tmp_path / "tmp" and q.name.startswith("collect_persist_gpu0_")
    assert not (tmp_path / "shared").exists()


def test_render_only_reader_has_no_sha_gate():
    """DEXENGINE_RENDER_ONLY_SHA is retired: the release reader carries no sha list check."""
    from dexengine.record.scene import render_only as ROL

    assert not any(hasattr(ROL, n) for n in ("SHA_ENV", "SHA_FILE", "_check_sha", "_expected", "sha_rows", "local_env"))


@pytest.mark.parametrize(
    "line",
    [
        "[COLLECT] --clips ['a'] are not in the judge record",
        "[COLLECT] DRYRUN failed rc 2: /x/dry.log",
    ],
)
def test_reason_of_finds_failure_lines(tmp_path, line):
    from dexengine.record import driver_check as DC

    log = tmp_path / "collect.log"
    log.write_text(
        "\n".join(
            ["[COLLECT] stage 2 present c (fingerprint f)", line, "[COLLECT] rollouts 64: chosen 3/3 · 0 successes []"]
        )
        + "\n"
    )
    assert DC.reason_of(log, 1) == f"rc 1 · {line}"


def test_owner_markers_checked_by_the_same_constants():
    from dexengine.record import sources as OWN

    src = f"{OWN.MARK} src=/x sha=a canon_sha=a"
    ok = f"{OWN.OWNER_LINE}/x sha abc · clips 2 · no copy"
    assert OWN.log_problems("\n".join([src, ok])) == []
    assert len(OWN.log_problems(src)) == 1 and len(OWN.log_problems(ok)) == 1


def test_env_fingerprint_reads_the_replay_env_package():
    from dexengine.record import cache as CA

    assert len(list(REPO.glob(CA.ENV_CODE_GLOB))) > 0
    assert len(CA.env_fingerprint(REPO)) == 16


def test_session_finish_and_arm_base_resolve():
    from dexengine.record.render import episode as EP
    from dexengine.record.store import scene_episode as SL

    assert callable(EP.SR.finish)
    assert SL._base_pos("rby1") == (-0.7, 0.0, 0.0)


def _bundle_rc_of_child_log(tmp_path, child_text: str) -> int:
    """A failed bundle whose collect log points at a child session log: driver_check.reason_of builds the reason the
    driver stores, driver_spec.bundle_rc classifies it."""
    from dexengine.record import driver_check as DC
    from dexengine.record import driver_spec as DS

    sess = tmp_path / "session.log"
    sess.write_text(child_text)
    log = tmp_path / "collect.log"
    log.write_text(f"[COLLECT] stage 1 c\n[COLLECT] ✗ the session died: see {sess}\n")
    return DS.bundle_rc({"state": "FAILED", "reason": DC.reason_of(log, 1)})


def _traceback_of(exc: BaseException) -> str:
    import traceback

    try:
        raise exc
    except BaseException as e:  # noqa: BLE001 — the text a session prints with traceback.print_exc
        return "".join(traceback.format_exception(e))


def test_input_missing_error_in_a_session_log_is_rc_12(tmp_path):
    from dexengine.core.errors import InputMissingError
    from dexengine.record import driver_spec as DS

    tb = _traceback_of(InputMissingError("[COLLECT_TREE] ✗ /a/tree_ref.json is missing: which tree …"))
    assert _bundle_rc_of_child_log(tmp_path, "Isaac boot\n" + tb) == DS.CODE["INPUT_MISSING"]


def test_input_missing_cross_line_without_exception_is_rc_12(tmp_path, capsys):
    from dexengine.core.errors import input_missing
    from dexengine.record import driver_spec as DS

    print(input_missing("[COLLECT_TREE] ✗ /a/tree_ref.json is missing"))
    assert _bundle_rc_of_child_log(tmp_path, "boot\n" + capsys.readouterr().out) == DS.CODE["INPUT_MISSING"]


@pytest.mark.parametrize(
    "exc",
    [
        SystemExit("[COLLECT_TREE] ✗ /a/tree_ref.json is missing: which tree …"),
        RuntimeError("[AUG_COLL] --no-ref with pv00 rollouts: one of the two"),
    ],
)
def test_other_failures_stay_broken(tmp_path, exc):
    """Only the marker counts: a plain SystemExit that says 'missing' is rc 1 (BROKEN)."""
    from dexengine.record import driver_spec as DS

    assert _bundle_rc_of_child_log(tmp_path, _traceback_of(exc)) == DS.BROKEN


def test_retarget_camera_matches_the_scene_camera():
    """viz.mano_model.RETARGET_CAM_XYAXES equals camera 'front' of the scene builder (dexengine/retarget/scene); reads
    the sources by AST (mano_model imports smplx/torch, so the value is read from its source too)."""
    import ast

    def call_kw(path, func, name):
        for n in ast.walk(ast.parse((REPO / path).read_text())):
            if (
                isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == func
                and any(k.arg == "name" and getattr(k.value, "value", None) == name for k in n.keywords)
            ):
                return {k.arg: ast.literal_eval(k.value) for k in n.keywords if k.arg in ("xyaxes", "pos")}
        return None

    src = next(
        (
            r
            for f in sorted((REPO / "dexengine/retarget/scene").rglob("*.py"))
            if (r := call_kw(str(f.relative_to(REPO)), "add_camera", "front"))
        ),
        None,
    )
    assert src is not None, "the scene builder (dexengine/retarget/scene) declares no `front` camera"
    body = ast.parse((REPO / "dexengine/record/viz/mano_model.py").read_text()).body
    mine = next(
        ast.literal_eval(n.value)
        for n in body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "RETARGET_CAM_XYAXES"
    )
    assert [float(x) for x in src["xyaxes"]] == [float(x) for x in mine]


@pytest.mark.parametrize("diag", ["{m}4{w}/x/diag.json"])
def test_augment_pod_log_verdict_reads_the_session_diag_line(diag):
    from dexengine.record.augment import run as AR
    from dexengine.record.augment import session as AS

    bundle = "[AUG_RUN] DONE ok=True rc=0 session 12.0 s · [AUG_PLACE] 2/2 · verdict ok · /x\n"
    sess = (
        "[AUG_PLACE] a\n[AUG_PLACE] b\n[TRAJ_REC] DONE E = E' ✓ · file /x/rollouts.npz · ok\n"
        + diag.format(m=AS.DIAG_MARK, w=AS.DIAG_WRITTEN)
        + "\n"
    )
    ok, kind = AR.pod_log_verdict(bundle, sess)
    assert ok and kind.startswith("ok · ")
    assert not AR.pod_log_verdict(bundle, sess.replace("/x/diag.json", "").replace(AS.DIAG_MARK, "x"))[0]


def test_cells_pick_done_line_is_what_augment_cells_checks():
    """augment.cells accepts the cell choice only when cells_pick's end line names the clip count."""
    import inspect

    from dexengine.record.augment import cells_pick as ACP  # augment.cells is read as text (it imports rl)

    assert "ACP.DONE_HEAD}{len(clips)} · " in (REPO / "dexengine/record/augment/cells_steps.py").read_text()
    assert "{DONE_HEAD}{len(regs)} · " in inspect.getsource(ACP.main)
    assert ACP.DONE_HEAD.startswith(ACP.MARK + " DONE")


def test_receiver_clearance_self_test_passes(capsys):
    """augment.recv_clearance --selftest on synthetic boxes: planted depths split the bands, removing them clears
    them."""
    from dexengine.record.augment import recv_clearance as RC

    assert RC.selftest() == 0


def test_visual_augment_reasons_keep_their_stored_text():
    """size_why and allowed() reasons reach the asset manifest and render meta, so their text stays as recorded."""
    from dexengine.record.looks.augvis import catalog as AC

    assert AC.size_why([0.5, 0.1, 0.1]) == "size [0.5, 0.1, 0.1] m > cap (height 0.3 · floor diagonal 0.48)"
    assert AC.size_why([0.01, 0.01, 0.01]).startswith("size [0.01, 0.01, 0.01] m < floor (")


@pytest.mark.parametrize(
    "reason,want",
    [
        ("rc 1 · [AUG] ✗ seed x ⇐ SystemExit: [TAG] [INPUT_MISSING] no tree", "INPUT_MISSING"),  # root line marked
        ("rc 1 · SystemExit: [cells] no canonical cell /x", "BROKEN"),  # unmarked English: not input missing
        ("rc 1 · SystemExit: [cells] [INPUT_MISSING] no canonical cell /x", "INPUT_MISSING"),
    ],
)
def test_bundle_rc_reads_stored_reasons_old_and_new(reason, want):
    from dexengine.record import driver_spec as DS

    rc = DS.bundle_rc({"state": "FAILED", "reason": reason, "rc": 1})
    assert rc == (DS.BROKEN if want == "BROKEN" else DS.CODE[want])
