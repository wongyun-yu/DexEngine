"""Identity record of derived robot USDs — the single owner of the file name, format and check.

A derived USD is a sim-only post-processed body of a generation's USD (``arms.RobotGeneration.derived_from``): same
kinematics, mass and visuals as its source, only the colliders differ. When the source is re-baked the derivative is
stale; so that a stale body is never spawned silently, the generator writes the source md5 into this record and the
pre-spawn check (``robots.selection.assert_generation_coherent``) compares it with the current source. A changed
collider definition also makes the derivative stale (under the same name), so the generator records the arm collider
digest (``sim_collider.digest``) and the check re-measures it with the sim_collider of THIS code tree. The generator
code (``GENERATOR_FILES``) is covered the same way: the digest of its bytes is recorded and compared with this tree.
A derivative baked from a material-only body (render body, ``RobotGeneration.material_of``) also records the md5 of
its material source (the training body) USD: the render-body generator leaves no record, so re-baking the training
body makes the render body stale here.
Imports without Kit (the launcher gate, the generator and CPU checks use the same functions).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

from dexengine.core.repo import REPO_ROOT

#: Name of the record file inside a derived USD folder (written by the generator, read by the check).
PROVENANCE_NAME = "derived_provenance.json"
#: Read the record from this folder (``<folder>/<derived dir name>/derived_provenance.json``) instead of next to the
#: derived USD. Explicit input only — set for the Isaac process by ``dexengine/rl/judge/rejudge.py --bundle-data``
#: (a re-judge on the training commit's tree boots with the training bundle's identity records). Unset = the record
#: next to the USD. Set but the variant's record missing = error (no fallback to the record next to the USD).
PROVENANCE_DIR_ENV = "DEXENGINE_DERIVED_PROVENANCE_DIR"

#: The code tree this file belongs to — both the generator digest and the current collider digest are measured here.
_REPO = REPO_ROOT
#: Record field of the arm collider digest (written by the generator, read by the check).
DIGEST_KEY = "sim_collider_digest"
#: "Digest measured now" marker — the sim_collider CLI prints it on stderr with the measuring tree and commit and the
#: caller relays it. Owned here (sim_collider reads it from this module).
DIGEST_MARK = "[COLLIDER_DIGEST]"
#: The one "digest now" entry: the sim_collider CLI (``--digest <arm> --urdf-of <hand> <body variant>`` = the arm shapes
#: of that one URDF; stderr carries the [COLLIDER_DIGEST] tree/commit marker). The judge side
#: (``sim_collider.measured_digest``) runs the same CLI.
_DIGEST_MODULE = "dexengine.placement.collision.sim_collider"
#: Generator code of the derived USDs (repository-relative, this tree) — the digest of these bytes is the record field
#: GENERATOR_KEY. The split body (_split) is made by ``split_arm.py``, which the generator calls, and has no record
#: of its own (the spawn check reads the base body's record), so a change of the split generator also stales the base
#: body's record and both are re-baked together. Each module carries its split-off part (collider_edit: the USD edits;
#: split_arm_check: the split's checks) — a generator module left off this list would change no digest.
GENERATOR_FILES = (
    "dexengine/robots/usd/collider_rev.py",
    "dexengine/robots/usd/collider_edit.py",
    "dexengine/robots/wrist_hull.py",
    "dexengine/robots/usd/split_arm.py",
    "dexengine/robots/usd/split_arm_check.py",
)
GENERATOR_KEY = "generator_digest"
#: Render-body record field — md5 of the material source (training body) USD.
MATERIAL_SOURCE_KEY = "material_source_md5"
#: Re-bake command, printed ready to paste when a record is stale (the interpreter comes from the paths CLI).
REBAKE_CMD = (
    '"$(python3 dexengine/core/paths.py --what retarget-py)" dexengine/robots/usd/collider_rev.py '
    "--arm {arm} --hand {hand} --force   # then the same line with --verify-only, on every machine that runs it"
)

_MD5_CACHE: dict[tuple[str, int, int], str] = {}
_DIGEST_CACHE: dict[tuple[str, str, str], str] = {}
_DIGEST_WHERE: dict[tuple[str, str, str], str] = {}  # the [COLLIDER_DIGEST] line of the same key (tree, commit)
_DIGEST_SAID: set[str] = set()


def provenance_file(usd_path: str | os.PathLike) -> pathlib.Path:
    """Record file of a derived USD — next to it, or under the same folder name in the folder named by
    PROVENANCE_DIR_ENV."""
    d = os.environ.get(PROVENANCE_DIR_ENV, "")
    if d:
        return pathlib.Path(d) / pathlib.Path(usd_path).parent.name / PROVENANCE_NAME
    return pathlib.Path(usd_path).parent / PROVENANCE_NAME


def file_md5(path: str | os.PathLike) -> str:
    """md5 of a file (read once per process for the same size and modification time)."""
    p = pathlib.Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _MD5_CACHE[key] = h.hexdigest()
    return _MD5_CACHE[key]


#: Identity-gate marker — one line comparing the spawned body's md5 with its expected value (printed right before
#: spawning by robots.selection.assert_generation_coherent).
PIN_TAG = "[HAND_USD]"
#: Expected md5 of spawned USDs (USD file stem → md5): bodies re-baked under the SAME name after a bake fix. A stale
#: copy with the same name (another machine's cache, a place that was not overwritten) could otherwise spawn silently,
#: so a body named here spawns only with this md5 (otherwise it stops). Bodies not listed are reported as unchecked.
#: After a re-bake, update the value with the product's md5 (measured with file_md5).
#: wuji2: re-baked after the URDF material rule (dexengine/robots/urdf/materials.py) — Isaac's import had dropped the
#: named inline colour materials of the merged URDF; outside materials the body equals the old one.
#: Render bodies (_mi): hand visuals repainted with the hand look declarations (dexengine/core/hand_looks.py); the
#: training bodies are byte-identical.
USD_MD5_PINS: dict[str, str] = {
    "rby1_wuji2_truncated": "33baee72c249c6ebd33a7b4f013d3291",
    "rby1_wuji2_truncated_armhull": "d8a80c69366ebd2f1bb4d403ae2780a0",
    "rby1_wuji2_truncated_mi": "64a9b3699139bf07e8789a3ce9de9e76",
    "rby1_wuji2_truncated_mi_armhull": "be5c2b905f9b18235eea9be239375309",
    "rby1_sharpa_wave_mi": "859f07b6d35cf5f92f35fe5ed91b63d6",
    "rby1_sharpa_wave_mi_armhull": "5079411d524e87d94a3d1228f867f7cb",
    "rby1_dg5f_s_truncated_mi": "06b05c73ab6e43835b1895a07d7c422e",
    "rby1_dg5f_s_truncated_mi_armhull": "bf53483040c0cbb5b5c6377acd594e7e",
}


def check_pin(usd_path: str | os.PathLike) -> str:
    """Identity gate of a spawned body: a body named in USD_MD5_PINS must have that md5 (otherwise RuntimeError).
    Returns one marker line. The line's text is recorded per episode by the recorder (``pin`` field) and read by the
    dataset merge gate, so it stays byte-identical."""
    p = pathlib.Path(usd_path)
    got, want = file_md5(p), USD_MD5_PINS.get(p.stem)
    # The two returned lines below are recorded data (kept verbatim): "no expected value (not checked)" / "= expected".
    if want is None:
        return f"{PIN_TAG} {p.stem} md5={got} · no expected value (not compared — derived_usd.USD_MD5_PINS)"
    if got != want:
        raise RuntimeError(
            f"{PIN_TAG} ✗ {p.stem} md5={got} ≠ expected {want} — a stale copy (a body from before a bake fix, same "
            f"name).\n  USD: {p}\n  Replace it with the canonical body (the md5 in derived_usd.USD_MD5_PINS) — the old "
            "body is not spawned."
        )
    return f"{PIN_TAG} {p.stem} md5={got} = expected"


def generator_digest(repo: str | os.PathLike | None = None) -> str:
    """Digest of the generator files' bytes (GENERATOR_FILES, with their repository-relative paths) — written into the
    record by the generator, compared with this tree's value by the spawn check. Files are read, not imported (no pxr
    or trimesh in the spawn process). A missing file is an error."""
    root, h = pathlib.Path(repo) if repo is not None else _REPO, hashlib.sha256()
    for rel in GENERATOR_FILES:
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return h.hexdigest()[:16]


def baked_bodies(arm: str, hands=None) -> list[tuple[str, str, str]]:
    """(hand, source body variant, derived variant) of every derivative the generator bakes for `arm` — each hand's two
    bodies (arms.hand_bodies) with the arm's collider-revision suffix (arms.ARM_SIM_COLLIDER_REV). `hands` default =
    every hand registered with the arm (sim_collider.arm_hands). The job list of dexengine/robots/usd/collider_rev.py
    and the bodies the legacy-data migration re-stamps."""
    from dexengine.core import arms as AS  # noqa: PLC0415 — only the generator and the migration need the tables
    from dexengine.placement.collision import sim_collider as SC  # noqa: PLC0415

    suffix = AS.ARM_SIM_COLLIDER_REV[arm].suffix
    return [(h, bv, bv + suffix) for h in (hands or SC.arm_hands(arm)) for bv in AS.hand_bodies(arm, h)]


def restamp_generator(text: str, old: str, new: str) -> str:
    """The record text with the GENERATOR_KEY value `old` replaced by `new` — only those bytes change (a JSON re-dump
    could change others). The value must occur exactly once and read back as `new`, else ValueError."""
    rx = re.compile(r'("' + re.escape(GENERATOR_KEY) + r'"\s*:\s*")' + re.escape(old) + '"')
    out, n = rx.subn(lambda m: m.group(1) + new + '"', text)
    if n != 1 or json.loads(out).get(GENERATOR_KEY) != new:
        raise ValueError(
            f"[derived_usd] {GENERATOR_KEY} {old!r} occurs {n} times in the record (exactly one is re-stamped)"
        )
    return out


def read(usd_path: str | os.PathLike) -> dict:
    """Read the record. Missing = loud error: a derivative without a record cannot prove its identity."""
    f = provenance_file(usd_path)
    named = os.environ.get(PROVENANCE_DIR_ENV, "")
    if not f.is_file():
        if named:
            raise RuntimeError(
                f"[derived_usd] {PROVENANCE_DIR_ENV}={named} but this variant has no identity "
                f"record there: {f} — the record next to the USD is not used instead"
            )
        raise RuntimeError(
            f"[derived_usd] the derived USD has no identity record: {f}\n"
            f"  Only the generator (dexengine/robots/usd/collider_rev.py) makes this body — a hand-copied body is not "
            "used. Re-bake it."
        )
    if named and str(f) not in _DIGEST_SAID:
        _DIGEST_SAID.add(str(f))
        print(
            f"[DERIVED_USD] identity record = {f} ({PROVENANCE_DIR_ENV} set — instead of the record next to the "
            "derived USD)",
            flush=True,
        )
    return json.loads(f.read_text(encoding="utf-8"))


def collider_digest_now(arm: str, hand: str, body_variant: str) -> str:
    """The arm collider digest this code tree's sim_collider gives for the (arm, hand, body variant) URDF — measured
    once per process, in a child process (the retargeting interpreter's mesh libraries are not needed in the spawn
    process). Failure to measure is an error: no spawn without the check."""
    key = (arm, hand, body_variant)
    if key not in _DIGEST_CACHE:
        try:
            r = subprocess.run(
                [sys.executable, "-m", _DIGEST_MODULE, "--digest", arm, "--urdf-of", hand, body_variant],
                cwd=_REPO,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                f"[derived_usd] cannot measure the arm collider digest ({arm}_{hand}{body_variant}): {e}"
            ) from e
        out = r.stdout.strip().splitlines()
        mark = [ln.strip() for ln in (r.stderr or "").splitlines() if ln.startswith(DIGEST_MARK)]
        if r.returncode != 0 or not out or not mark:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            raise RuntimeError(
                f"[derived_usd] cannot measure the arm collider digest ({arm}_{hand}{body_variant}, "
                f"rc={r.returncode}, {_DIGEST_MODULE}): {tail}"
            )
        _DIGEST_CACHE[key], _DIGEST_WHERE[key] = out[-1].strip(), mark[-1]
    return _DIGEST_CACHE[key]


def mismatches(
    prov: dict,
    variant: str,
    source_variant: str,
    source_usd: str,
    *,
    arm: str,
    hand: str,
    material_source_usd: str | None,
    generator: bool = True,
) -> tuple[list[str], str | None, str]:
    """The identity comparison of one derived-USD record `prov` → (mismatch lines, the collider digest measured now
    (None when the record has none), this tree's generator digest). The one rule behind `verify` (spawn check) and a
    generator re-stamp, which passes generator=False: everything
    else — variant pair, source md5, arm collider digest measured with THIS tree's sim_collider, render-body material
    source — must still match before the generator digest may be re-stamped."""
    bad = []
    if prov.get("variant") != variant:
        bad.append(f"recorded variant {prov.get('variant')!r} ≠ spawned variant {variant!r}")
    if prov.get("source_variant") != source_variant:
        bad.append(
            f"recorded source variant {prov.get('source_variant')!r} ≠ generation table derived_from {source_variant!r}"
        )
    if not pathlib.Path(source_usd).is_file():
        bad.append(f"source USD missing: {source_usd}")
    elif prov.get("source_md5") != file_md5(source_usd):
        bad.append(
            f"the source USD changed after deriving (recorded md5 {prov.get('source_md5')} ≠ now "
            f"{file_md5(source_usd)}) — the derivative is stale, run the generator again"
        )
    gen = generator_digest()
    if generator and GENERATOR_KEY not in prov:
        bad.append(f"the record has no {GENERATOR_KEY} — the generator code that baked it cannot be proven")
    elif generator and prov[GENERATOR_KEY] != gen:
        bad.append(
            f"the generator code changed after deriving (recorded {prov[GENERATOR_KEY]} ≠ now {gen} · "
            f"{' · '.join(GENERATOR_FILES)}) — the derivative is stale"
        )
    rec, now = prov.get(DIGEST_KEY), None
    if rec is None:
        bad.append(f"the record has no {DIGEST_KEY} — the collider definition it was baked with cannot be proven")
    else:
        now = collider_digest_now(arm, hand, source_variant)
        if rec != now:
            bad.append(
                f"the arm collider definition changed after deriving (recorded {rec} ≠ now sim_collider "
                f"{now}) — the derivative is stale"
            )
    if (MATERIAL_SOURCE_KEY in prov) != (material_source_usd is not None):
        has = "present" if MATERIAL_SOURCE_KEY in prov else "absent"
        bad.append(
            f"render-body state disagrees — record {MATERIAL_SOURCE_KEY} {has} · "
            f"material source passed to the check {material_source_usd or 'none'}"
        )
    elif material_source_usd is not None and not pathlib.Path(material_source_usd).is_file():
        bad.append(f"the render body's material source (training body) USD is missing: {material_source_usd}")
    elif material_source_usd is not None and prov[MATERIAL_SOURCE_KEY] != file_md5(material_source_usd):
        bad.append(
            f"the training body USD changed after the render body (recorded {MATERIAL_SOURCE_KEY} "
            f"{prov[MATERIAL_SOURCE_KEY]} ≠ now {file_md5(material_source_usd)}) "
            "— remake the render body (dexengine/robots/usd/render_body.py) first"
        )
    return bad, now, gen


def verify(
    usd_path: str,
    variant: str,
    source_variant: str,
    source_usd: str,
    *,
    arm: str,
    hand: str,
    material_source_usd: str | None,
) -> dict:
    """Check that a derived USD pairs with (its variant, the source variant, the current source file, this tree's
    generator, the current arm collider definition and, for a render body, the current material source file).

    ``material_source_usd`` = the training body USD of a render body, None otherwise — always stated by the caller; a
    disagreement with the record's render-body state is an error. A mismatch raises RuntimeError (with the re-bake
    command); a match prints one ``[DERIVED_USD_DIGEST]`` line."""
    prov = read(usd_path)
    bad, now, gen = mismatches(
        prov, variant, source_variant, source_usd, arm=arm, hand=hand, material_source_usd=material_source_usd
    )
    if bad:
        raise RuntimeError(
            f"[derived_usd] derived USD identity mismatch — {usd_path}\n  "
            + "\n  ".join(bad)
            + "\n  re-bake: "
            + REBAKE_CMD.format(arm=arm, hand=hand)
        )
    if str(usd_path) not in _DIGEST_SAID:
        _DIGEST_SAID.add(str(usd_path))
        print(
            f"[DERIVED_USD_DIGEST] {variant} arm collider digest {now} = "
            f"sim_collider now ({arm}_{hand}{source_variant}) — {_DIGEST_WHERE.get((arm, hand, source_variant))}"
            f" · generator {gen}",
            flush=True,
        )
    return prov
