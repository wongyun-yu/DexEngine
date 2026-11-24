"""Training robot USD of a merged combo: the plan, its input check and the step runner.

One combo asset folder `<dir>` (the combo id plus the arm's generation suffix, e.g. `rby1_sharpa_wave`, or a
variant folder such as `<dir>_truncated`) is built from the merged URDF `assets/robots/full_robot_gen/<dir>/<dir>.urdf`
into `<data root>/data/usd_cache/robots/full_robot_gen/<dir>/`:

  1 convert    dexengine/rl/assets/urdf_to_usd.py       URDF → <dir>_base_test.usd   (Isaac Lab URDF importer, Kit)
  2 flatten    dexengine/robots/usd/flatten.py          → <dir>_flat_test.usd        (pxr)
  3 export     dexengine/robots/usd/mimic.py            → <dir>_stage3_test.usd      (pxr, single layer)
  4 reporters  dexengine/rl/assets/contact_report_bake.py  every combo in one Kit boot → <dir>.usd (canonical name)
  5 colours    dexengine/robots/usd/check_colors.py     URDF visual colours ↔ USD materials

A canonical folder's `<dir>.usd` is the training USD: its path relative to `data/usd_cache/robots/` must equal the
pair's `paths.combo_usd_relpath` (checked in the plan). Every step runs with the Isaac Lab python
(`scripts/_isaac_env.sh` activates it; `<IsaacLab>/isaaclab.sh -p`), the code tree on PYTHONPATH and the data root in
DEXENGINE_MAIN_ROOT.

Done = counts on the outputs, not exit codes: per combo the stage-3 file, the `[stage4] <dir>: … OK` line, a canonical
`<dir>.usd` written by this run and a passing colour check. The last line is the marker `COMBO_USD_DONE stage3 k/n ·
stage4 rc=… · canonical usd k/n · colour check k/n`. The plan and `check_inputs` start no Isaac process and write
nothing (the driver's `--dry-run`).

CLI: `python scripts/build_robot_usd.py` (see its --help).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.repo import CONFIGS_DIR, FULL_ROBOT_URDF_DIR, REPO_ROOT

#: Robot USD folder under the data root (paths.combo_usd_relpath is relative to it).
ROBOT_USD_REL = "data/usd_cache/robots"
#: The merged-combo subfolder of both the URDF assets and the USD cache.
GEN_SUBDIR = "full_robot_gen"
CONVERTER_CONFIG = CONFIGS_DIR / "robots" / "urdf_converter.yaml"
#: (step, script) in run order; step 4 runs once for every combo.
STEPS = (
    ("convert", "dexengine/rl/assets/urdf_to_usd.py"),
    ("flatten", "dexengine/robots/usd/flatten.py"),
    ("export", "dexengine/robots/usd/mimic.py"),
    ("reporters", "dexengine/rl/assets/contact_report_bake.py"),
    ("colours", "dexengine/robots/usd/check_colors.py"),
)
SCRIPT = dict(STEPS)
#: Intermediate file suffix per step (the canonical name `<dir>.usd` is step 4's).
_SUFFIX = {"convert": "_base_test.usd", "flatten": "_flat_test.usd", "export": "_stage3_test.usd"}
#: How a step runs: activate the Isaac Lab venv (EULA, ISAACLAB_DIR) and hand the script to its launcher.
_ISAAC_RUN = 'source scripts/_isaac_env.sh >&2 || exit 2; exec "$ISAACLAB_DIR/isaaclab.sh" -p "$@"'


@dataclass(frozen=True)
class ComboPlan:
    """One combo asset folder: inputs and outputs of the five steps."""

    dir: str
    arm: str
    hand: str
    variant: str
    urdf: Path
    usd_dir: Path
    canonical: bool  # dir == paths.combo_dirname(<arm>_<hand>): the output is the training USD

    def out(self, step: str) -> Path:
        return self.usd_dir / (f"{self.dir}.usd" if step == "reporters" else f"{self.dir}{_SUFFIX[step]}")


def usd_root(data_root: Path | None = None) -> Path:
    """`<data root>/data/usd_cache/robots` (the data root of this process when not given)."""
    return Path(data_root or P.main_root()) / ROBOT_USD_REL


def default_dirs(arm: str | None = None, hand: str | None = None) -> list[str]:
    """The canonical asset folders of every registered combo (paths combo-dirs), or of one arm and/or hand. A hand
    whose vendor source is not set (``vendor.unavailable_hands``) is left out with a ``[HANDS]`` line, and asking for
    it by name dies."""
    from dexengine.robots import vendor  # noqa: PLC0415

    un = vendor.unavailable_hands()
    if hand in un:
        raise SystemExit(f"[HANDS] hand {hand} not available: {un[hand]}")
    out, left = [], []
    for c in P.combo_ids():
        a, h, _v = P.split_combo(c, P.ARM_IDS, P.HAND_IDS)
        if (arm is None or a == arm) and (hand is None or h == hand):
            (left if h in un else out).append(P.combo_dirname(c))
    if left:
        print(f"[HANDS] combos selected {out} · not selected {left} (vendor source not set)", flush=True)
    if not out:
        raise ValueError(f"no registered combo for arm={arm!r} hand={hand!r} (combos: {list(P.combo_ids())})")
    return out


def plan(dirs: list[str], data_root: Path | None = None) -> list[ComboPlan]:
    """The plan of each asset folder. An unknown folder name (not <arm>_<hand>[<variant>]) raises ValueError."""
    root = usd_root(data_root) / GEN_SUBDIR
    out = []
    for d in dirs:
        arm, hand, var = P.split_combo(d, P.ARM_IDS, P.HAND_IDS)
        canon = d == P.combo_dirname(f"{arm}_{hand}")
        out.append(ComboPlan(d, arm, hand, var, FULL_ROBOT_URDF_DIR / d / f"{d}.urdf", root / d, canon))
    return out


def _sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def mesh_report(urdf: Path) -> dict:
    """The URDF's mesh references as the two readers see them: the resolver (paths.urdf_mesh — tracked meshes from this
    code tree, the vendor stages from the data root) and Isaac's importer (relative to the URDF, as written). The
    importer reads neither the resolver nor the data root, so the two must name the same existing file."""
    refs = [m.get("filename") or "" for m in ET.parse(urdf).getroot().iter("mesh")]
    missing, split = [], []
    for fn in refs:
        own = P.urdf_mesh(urdf, fn)
        lex = own if "://" in fn else (urdf.parent / fn).resolve()
        if not own.is_file():
            missing.append(fn)
        elif own != lex:
            split.append(f"{fn} -> importer {lex} · resolver {own}")
    return {"n": len(refs), "files": len(set(refs)), "missing": missing, "split": split}


#: The repo's bake record next to a body (written by ``run`` when the body's canonical USD is done): it binds the
#: converter's own record (``.asset_hash`` — Isaac Lab's, read here, never written) to the USD bytes the bake produced.
BAKE_RECORD = "bake_record.json"


def converter_hash(usd_dir: Path, urdf: Path) -> str:
    """Isaac Lab's converter record formula (isaaclab.sim.converters.asset_converter_base._config_to_hash): md5 of the
    json of its config (``config.yaml`` without the three path keys) + the URDF bytes."""
    import json  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    d = yaml.safe_load((Path(usd_dir) / "config.yaml").read_text(encoding="utf-8")) or {}
    for k in ("asset_path", "usd_dir", "usd_file_name"):
        d.pop(k, None)
    md5 = hashlib.md5(json.dumps(d).encode())
    md5.update(Path(urdf).read_bytes())
    return md5.hexdigest()


def write_bake_record(usd_dir: Path, urdf: Path, usd: Path) -> dict:
    """Bind the bake: the converter's ``.asset_hash`` (it must be the formula on these URDF bytes — a stale one dies),
    the URDF's sha256 and the name and md5 of the USD the bake produced, in ``BAKE_RECORD``. The URDF's path is the
    converter's (``config.yaml`` asset_path) and is not restated here."""
    import json  # noqa: PLC0415

    from dexengine.robots import derived_usd as DU  # noqa: PLC0415

    usd_dir, urdf, usd = Path(usd_dir), Path(urdf), Path(usd)
    ah = (usd_dir / ".asset_hash").read_text(encoding="utf-8").strip()
    if ah != converter_hash(usd_dir, urdf):
        raise RuntimeError(
            f"{usd_dir}/.asset_hash is not the converter record of {urdf} — the conversion did not rewrite it"
        )
    rec = {
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "asset_hash": ah,
        "usd": usd.name,
        "usd_md5": DU.file_md5(usd),
    }
    (usd_dir / BAKE_RECORD).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


#: ``baked_from`` answers when the USD is not the bytes the bake wrote (a later write left the records behind).
STALE = "record stale: the USD changed after the bake"


def baked_from(usd_dir: Path, urdf: Path, usd: Path | None = None) -> str:
    """Whether the body in ``usd_dir`` is the bake of these URDF bytes, as ``BAKE_RECORD`` states it. "same" only when
    the URDF's sha256 is the recorded one, the converter record (``.asset_hash``) is the one the bake bound and the
    formula on this config and URDF still gives it, and the USD's md5 is the recorded one (``usd``, when given, must be
    the USD the record binds). Otherwise "no record" · "not the baked USD" · "other URDF bytes" · "converter record
    changed" · "converter config changed" · ``STALE``.
    Readers: ``built_from`` (the plan report) and ``robots.usd.head_camera.body_urdf`` (the render body reads the
    camera visual from that URDF only when "same")."""
    import json  # noqa: PLC0415

    from dexengine.robots import derived_usd as DU  # noqa: PLC0415

    usd_dir, urdf = Path(usd_dir), Path(urdf)
    files = [usd_dir / BAKE_RECORD, usd_dir / ".asset_hash", usd_dir / "config.yaml", urdf]
    if not all(f.is_file() for f in files):
        return "no record"
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    if usd is not None and Path(usd).name != rec.get("usd"):
        return "not the baked USD"
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != rec.get("urdf_sha256"):
        return "other URDF bytes"
    if rec.get("asset_hash") != files[1].read_text(encoding="utf-8").strip():
        return "converter record changed"
    if converter_hash(usd_dir, urdf) != rec["asset_hash"]:
        return "converter config changed"
    usd = usd_dir / str(rec.get("usd", ""))
    if not usd.is_file() or DU.file_md5(usd) != rec.get("usd_md5"):
        return STALE
    return "same"


def built_from(p: ComboPlan) -> str:
    """``baked_from`` for a plan: its USD folder and its URDF."""
    return baked_from(p.usd_dir, p.urdf)


def check_inputs(plans: list[ComboPlan]) -> tuple[list[str], list[str]]:
    """(one report line per combo, problems). A problem stops a real run before any Isaac process starts."""
    lines, bad = [], []
    if not CONVERTER_CONFIG.is_file():
        bad.append(f"converter config missing: {CONVERTER_CONFIG}")
    bad.extend(
        f"step script missing: {s}" for s in dict.fromkeys(s for _st, s in STEPS) if not (REPO_ROOT / s).is_file()
    )
    for p in plans:
        if not p.urdf.is_file():
            bad.append(
                f"{p.dir}: merged URDF missing: {p.urdf} (python dexengine/robots/urdf/merge.py {p.arm}+{p.hand})"
            )
            continue
        if p.urdf.name.endswith("_render.urdf"):
            bad.append(f"{p.dir}: a render URDF is not a bake input")
        m = mesh_report(p.urdf)
        bad += [f"{p.dir}: mesh missing: {f}" for f in m["missing"][:5]]
        bad += [
            f"{p.dir}: importer reads another file than the resolver: {s} — bake with the data root = the code "
            "tree (no DEXENGINE_MAIN_ROOT / --data-root)"
            for s in m["split"][:3]
        ]
        rel = p.out("reporters").relative_to(p.usd_dir.parents[1])  # relative to data/usd_cache/robots
        want = P.combo_usd_relpath(P.combo_id(p.arm, p.hand))
        if p.canonical and str(rel) != want:
            bad.append(f"{p.dir}: output {rel} ≠ paths.combo_usd_relpath {want} — the training reads the latter")
        kind = f"canonical (= paths.combo_usd_relpath {want})" if p.canonical else f"variant {p.variant!r}"
        have = f"present (converted from this URDF: {built_from(p)})" if p.out("reporters").is_file() else "absent"
        lines.append(
            f"[ROBOT_USD_PLAN] {p.dir} urdf={os.path.relpath(p.urdf, REPO_ROOT)} sha256={_sha12(p.urdf)} "
            f"meshes={m['n']} refs/{m['files']} files (missing {len(m['missing'])} · importer≠resolver "
            f"{len(m['split'])}) → {p.usd_dir} · {kind} · current USD {have}"
        )
    lines.append(
        f"[ROBOT_USD_PLAN] converter config {os.path.relpath(CONVERTER_CONFIG, REPO_ROOT)} "
        f"sha256={_sha12(CONVERTER_CONFIG) if CONVERTER_CONFIG.is_file() else '-'}"
    )
    return lines, bad


def commands(plans: list[ComboPlan]) -> list[tuple[str, str, list[str]]]:
    """(step, combo dir or '*', script argv) in run order."""
    out = []
    for p in plans:
        out += [
            (
                "convert",
                p.dir,
                [SCRIPT["convert"], "--input-urdf", str(p.urdf), "--output-usd", str(p.out("convert")), "--force"],
            ),
            (
                "flatten",
                p.dir,
                [SCRIPT["flatten"], "--input", str(p.out("convert")), "--output", str(p.out("flatten"))],
            ),
            (
                "export",
                p.dir,
                [SCRIPT["export"], "--input-usd", str(p.out("flatten")), "--output-usd", str(p.out("export"))],
            ),
        ]
    if plans:
        out.append(
            (
                "reporters",
                "*",
                [
                    SCRIPT["reporters"],
                    "--combos",
                    ",".join(p.dir for p in plans),
                    "--usd-root",
                    str(plans[0].usd_dir.parent),
                ],
            )
        )
    out += [
        ("colours", p.dir, [SCRIPT["colours"], "--urdf", str(p.urdf), "--usd", str(p.out("reporters"))]) for p in plans
    ]
    return out


def step_env(data_root: Path) -> dict:
    """The children's environment: this code tree importable, the data root named."""
    pp = os.environ.get("PYTHONPATH", "")
    return dict(
        os.environ, PYTHONPATH=str(REPO_ROOT) + (os.pathsep + pp if pp else ""), DEXENGINE_MAIN_ROOT=str(data_root)
    )


def _run(argv: list[str], log: Path, env: dict) -> int:
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.run(
            ["bash", "-c", _ISAAC_RUN, "_", *argv], cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT
        ).returncode


def _first_error(log: Path) -> str:
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"error|traceback", line, re.I):
            return line[:100]
    return "(no error line)"


def run(plans: list[ComboPlan], log_dir: Path, data_root: Path) -> int:
    """Run the five steps; print the per-combo lines and COMBO_USD_DONE; bind each finished body in ``BAKE_RECORD``.
    rc 0 only when every count is complete.
    The canonical `<dir>.usd` is not removed first: step 4 replaces it, and the count requires a file written now."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env, t0, n = step_env(data_root), time.time(), len(plans)
    ok = []
    for p in plans:
        p.usd_dir.mkdir(parents=True, exist_ok=True)
        for st in ("convert", "flatten", "export"):
            p.out(st).unlink(missing_ok=True)
    by_step = commands(plans)
    for p in plans:
        for st, d, argv in by_step:
            if d == p.dir and st in _SUFFIX:
                _run(argv, log_dir / f"{p.dir}_{list(_SUFFIX).index(st) + 1}{st}.log", env)
                if not p.out(st).is_file():
                    break
        if p.out("export").is_file():
            ok.append(p)
        else:
            last = max(log_dir.glob(f"{p.dir}_*.log"), key=os.path.getmtime)
            print(f"{p.dir}: no stage3 output — {last.name}: {_first_error(last)}", flush=True)
    s4log = log_dir / "stage4.log"
    s4 = _run(next(a for st, _d, a in by_step if st == "reporters"), s4log, env) if plans else 2
    s4lines = [x for x in s4log.read_text(encoding="utf-8", errors="replace").splitlines() if x.startswith("[stage4]")]
    print("\n".join(s4lines), flush=True)
    s4ok = {p.dir for p in plans if any(x.startswith(f"[stage4] {p.dir}: ") and x.endswith(" OK") for x in s4lines)}
    final = [
        p for p in plans if p.dir in s4ok and p.out("reporters").is_file() and p.out("reporters").stat().st_mtime >= t0
    ]
    col = recs = 0
    for st, d, argv in by_step:
        if st != "colours" or not any(p.dir == d for p in final):
            continue
        log = log_dir / f"{d}_5colors.log"
        rc = _run(argv, log, env)
        col += rc == 0
        p = next(p for p in final if p.dir == d)
        try:
            rec = write_bake_record(p.usd_dir, p.urdf, p.out("reporters"))
            print(f"[BAKE_RECORD] {d} asset_hash {rec['asset_hash'][:12]} usd_md5 {rec['usd_md5'][:12]}", flush=True)
            recs += 1
        except (RuntimeError, OSError) as e:  # a stale or missing converter record: counted, not a crash
            print(f"[BAKE_RECORD] {d} FAIL {e}", flush=True)
        print(
            next(
                (x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if "[VISUAL_COLORS]" in x),
                f"[VISUAL_COLORS] {d}: no marker (rc {rc})",
            ),
            flush=True,
        )
    print(
        f"COMBO_USD_DONE stage3 {len(ok)}/{n} · stage4 rc={s4} · canonical usd {len(final)}/{n} · "
        f"colour check {col}/{n} · bake record {recs}/{n}",
        flush=True,
    )
    return 0 if len(ok) == n and len(final) == n and s4 == 0 and col == n and recs == n else 1
