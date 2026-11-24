#!/usr/bin/env python3
"""HRDexDB desk height: sets desk_z for clips that start in a grasp (no placement); writes only with --write-intake (dry
by default).

Fact: the converter (dexengine/data/readers/hrdexdb.py) grounds every clip with "lowest object point at frame 0 = 0".
So the desk height in our frame is desk_z = −h0, with h0 = how far the frame-0 object bottom sits above the desk in the
source world. No transform t (Kabsch) is needed.

Modes (required, no default):
  h1p  policy: desk_z = 0 (frame-0 bottom) · desk_mode = table, the contract's "tabletop capture datasets are table
       (z 0) even without a placement". Needs no source. The h1 measurement below is the oracle checking this statement
       (fails when |0 − h1| = |h0| > H0_MAX_M).
  h1   measurement: desk_z = −h0, h0 = (that episode's frame-0 source bottom) − (median frame-0 source bottom over the
       same object's table clips). Objects without table clips are filled with the global median, only with an explicit
       --allow-global-level (no silent substitute).
Checks of the h1 measurement (both modes; any violation fails without writing):
  LOO   for objects with target clips, leave each table clip (truth = INTAKE measured placement desk_z) out and estimate
        it with the same rule; max |error| ≤ LOO_MAX_M
  h0    target clips |h0| ≤ H0_MAX_M (even grasp starts begin above the desk)
  hand  INTAKE hand_zmin ≥ desk_z − HAND_TOL (the hand never goes more than 2 cm below the desk)
Targets: clips of INTAKE support_audit.json with desk_mode ∈ {table_unknown, table_recovered}.
Version: each written clip gets desk_rule{version = RULE_VERSION, mode} (same form as the OakInk2 desk_rule;
    base_adapter copies it into the tree's support).
    INTAKE is written only with WRITE_MODE (rule D10-hrd option H1′ = h1p); h1 is oracle/dry only (the write point
        apply_updates fails).
Verification (--verify-base, read-only): measurement, verdict, marker and thresholds belong to
dexengine.data.support.hrdexdb_verify (only the entry is here; main hands over).
Usage: python -m dexengine.data.support.hrdexdb --mode h1p|h1 --out <dir> [--allow-global-level]
       [--write-intake (h1p only)]   ·
--selftest
       python -m dexengine.data.support.hrdexdb --verify-base --hand <hand> --dataset <ds> --root <data root>
       --out <dir>
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from dexengine.core import paths as P  # noqa: E402
from dexengine.core.errors import InputMissingError
from dexengine.data.readers import source_frames as SF  # noqa: E402  (the reader's camera-to-robot rule)
from dexengine.data.support.plane import (  # noqa: E402 — owner of the desk_mode vocabulary
    DESK_HEIGHT_UNKNOWN,
    DESK_RECOVERED,
    DESK_TABLE,
)

SRC_FPS, V_REST = 30.0, 0.03  # source 30 fps · rest speed (diagnostic scripts read the same values)
LOO_MAX_M, H0_MAX_M, HAND_TOL = 0.006, 0.010, 0.02
TARGET_MODES = (DESK_HEIGHT_UNKNOWN, DESK_RECOVERED)
# Marker of clips this tool wrote: removed from the truth (table measurements) and put back into the targets (idempotent
# reruns, independent oracle)
MARK = "desk_z_level_src"
#: Version of this rule, kept as desk_rule.version in every clip written to INTAKE. Value = decision date of D10-hrd
#: (H1′).
#: Bump it when modes, thresholds, targets or the marker change.
RULE_VERSION = "2026-10-02"
#: The only mode allowed to write INTAKE: rule D10-hrd option H1′ (h1p, tabletop capture policy desk_z 0).
#: h1 (measured h0 as desk_z) is a rejected alternative and runs only as oracle/dry (write attempts fail before
#: reading).
WRITE_MODE = "h1p"


def load_obj(p: Path, n=3000):
    with open(p, errors="replace") as fh:
        v = np.array([list(map(float, ln.split()[1:4])) for ln in fh if ln.startswith("v ")], dtype=float)
    if len(v) == 0:
        return None
    if (v.max(0) - v.min(0)).max() > 3.0:
        v = v / 1000.0  # mesh in mm
    return v[np.linspace(0, len(v) - 1, min(n, len(v))).astype(int)]


def episode_world_poses(ep: Path, c2r: np.ndarray):
    """Object poses (camera rig world) → robot base frame (+z up), the same transform as the reader (SF.robot_from_world
    =
    official inv(C2R), HRDexDB common.py:328-329) and the same pose file (SF.hrdexdb_pose_npz, one owner with the
    reader; old txt without it)."""
    f = SF.hrdexdb_pose_npz(ep)
    if f is not None:
        with np.load(f) as z:
            n = len([k for k in z.files if k.startswith("frame_")])
            mats = np.stack([z[f"frame_{i}"] for i in range(n)])
    else:
        mats = np.array([np.loadtxt(x) for x in sorted((ep / "object_6d").glob("pose_*.txt"))])
    return SF.robot_from_world(c2r)[None] @ mats


def frame0_floors(raw: Path) -> dict[str, float]:
    """Clip (<obj>__<ep>) → frame-0 object bottom z in the robot frame (all v1 mesh vertices). The C2R must be the one
    shared by the release (checked by SF.release_c2r; a difference fails), so that the single matrix fits every INTAKE
    clip even when the reader handled episodes without C2R through allow_shared_c2r."""
    try:
        c2r = SF.release_c2r(raw / "human")
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[hrdexdb_desk_recover] {e}") from e
    meshes, out = {}, {}
    for ep in sorted(raw.glob("human/*/*")):
        if not ((ep / "object_6d").is_dir() or SF.hrdexdb_pose_npz(ep) is not None):
            continue
        obj = ep.parent.name
        if obj not in meshes:
            m = SF.hrdexdb_mesh(raw, obj)
            if m is None:
                raise InputMissingError(f"source mesh missing: {obj}")
            meshes[obj] = load_obj(m, n=10**9)
        if meshes[obj] is None:
            raise InputMissingError(f"source mesh missing: {obj}")
        W0 = episode_world_poses(ep, c2r)[0]
        out[f"{obj}__{ep.name}"] = float((meshes[obj] @ W0[:3, :3].T + W0[:3, 3])[:, 2].min())
    return out


def obj_of(c: str) -> str:
    return c.split("__")[0]


def level(o: str, f0w: dict, table: list, excl: str | None = None) -> tuple[float, str, int]:
    """Desk level of object o (source world) = median frame-0 bottom over its table clips; without them (global median,
    'global_fill')."""
    v = [f0w[c] for c in table if obj_of(c) == o and c != excl]
    if v:
        return float(np.median(v)), "obj", len(v)
    g = [f0w[c] for c in table if c != excl]
    return float(np.median(g)), "global_fill", 0


def _loo_errors(sa: dict, f0w: dict, table: list, targets: list) -> list[float]:
    """Leave-one-out level error (m) of each reference table clip.

    Reference = table clips of objects with target clips (the range the oracle measured for the D10-hrd decision,
    when only those objects' sources were available). Objects without targets (e.g. the long-handled balloon_whisk,
    an 8.7 mm tail over all 421 public clips) do not bear on this rule's accuracy."""
    tobj = {obj_of(c) for c in targets}
    ref = [c for c in table if obj_of(c) in tobj] or table
    loo = [abs(-(f0w[c] - level(obj_of(c), f0w, table, excl=c)[0]) - float(sa[c]["desk_z"])) for c in ref]
    return loo


def _target_record(c: str, sa: dict, f0w: dict, table: list, mode: str, allow_global: bool, bad: list) -> dict:
    """New desk height of one target clip under the mode; rule violations are appended to bad."""
    lv, src, n = level(obj_of(c), f0w, table)
    if src == "global_fill" and mode == "h1" and not allow_global:
        bad.append(f"{c}: object {obj_of(c)} has 0 table clips — a global level needs an explicit --allow-global-level")
    h0 = f0w[c] - lv
    h1 = -h0
    new = 0.0 if mode == "h1p" else h1
    hz = sa[c].get("hand_zmin")
    if abs(h0) > H0_MAX_M:
        bad.append(f"{c}: |h0| {h0 * 1e3:+.1f} mm > {H0_MAX_M * 1e3:.0f}")
    if hz is not None and float(hz) < new - HAND_TOL:
        bad.append(f"{c}: hand lowest {float(hz) * 1e3:+.1f} mm < desk {new * 1e3:+.1f} − {HAND_TOL * 1e3:.0f}")
    return {
        "mode_before": sa[c].get("desk_mode"),
        "desk_z_before": sa[c].get("desk_z"),
        "desk_z": round(new, 5),
        "desk_z_h1": round(h1, 5),
        "h0": round(h0, 5),
        "level_src": src,
        "level_n": n,
        "delta_mm": round((new - float(sa[c].get("desk_z") or 0.0)) * 1e3, 1)
        if sa[c].get("desk_z") is not None
        else None,
    }


def decide(sa: dict, f0w: dict, mode: str, allow_global: bool) -> dict:
    """Apply the rule and check it. A violation raises SystemExit (fails before writing)."""
    if mode not in ("h1", "h1p"):
        raise InputMissingError(f"--mode {mode!r} — one of h1 | h1p (no default)")
    table = sorted(
        c
        for c, r in sa.items()
        if r.get("desk_mode") == DESK_TABLE and r.get("desk_z") is not None and MARK not in r and c in f0w
    )
    targets = sorted(c for c, r in sa.items() if r.get("desk_mode") in TARGET_MODES or MARK in r)
    if not table:
        raise InputMissingError("0 table clips — no truth to measure the level from")
    miss = [c for c in targets if c not in f0w]
    if miss:
        raise SystemExit(f"targets missing from the source {len(miss)}: {miss[:5]}")
    loo = _loo_errors(sa, f0w, table, targets)
    bad, clips = [], {}
    if max(loo) > LOO_MAX_M:
        bad.append(f"LOO max {max(loo) * 1e3:.1f} mm > {LOO_MAX_M * 1e3:.0f}")
    for c in targets:
        clips[c] = _target_record(c, sa, f0w, table, mode, allow_global, bad)
    if bad:
        raise SystemExit("checks failed — not writing:\n  " + "\n  ".join(bad))
    a = np.array(loo) * 1e3
    return {
        "mode": mode,
        "table_n": len(table),
        "targets_n": len(targets),
        "loo_mm": {
            "p50": round(float(np.median(a)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "max": round(float(a.max()), 2),
        },
        "clips": clips,
    }


def check_write_mode(mode: str) -> None:
    """INTAKE is written only with WRITE_MODE (D10-hrd H1′). Other modes raise SystemExit (main calls this even before
    reading sources)."""
    if mode != WRITE_MODE:
        raise SystemExit(
            f"--write-intake only with --mode {WRITE_MODE} (decision D10-hrd H1′) — {mode!r} is oracle/dry only"
        )


def apply_updates(clips: dict, res: dict) -> None:
    """Put the decided values and the version desk_rule{version, mode} onto INTAKE clips; the only place that changes
    INTAKE values (write_intake · self-test). Fails before changing anything unless the mode is WRITE_MODE."""
    check_write_mode(res["mode"])
    for c, r in res["clips"].items():
        clips[c].update(
            {
                "desk_z": r["desk_z"],
                "desk_z_h1": r["desk_z_h1"],
                "desk_z_h0": r["h0"],
                MARK: r["level_src"],
                "desk_mode": DESK_TABLE,
                "desk_z_confidence": "policy_f0_floor",
                "desk_rule": {"version": RULE_VERSION, "mode": res["mode"]},
            }
        )


def write_intake(ip: Path, res: dict) -> int:
    SA = json.loads(ip.read_text())
    apply_updates(SA["clips"], res)
    SA.setdefault("recovery", {})["hrdexdb_desk_recover"] = {
        "tool": "dexengine/data/support/hrdexdb.py",
        "mode": res["mode"],
        "rule_version": RULE_VERSION,
        "updated": len(res["clips"]),
        "loo_mm": res["loo_mm"],
    }
    tmp = ip.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(SA, ensure_ascii=False, indent=1, default=str))
    tmp.replace(ip)
    return len(res["clips"])


def _selftest_table() -> tuple[dict, dict]:
    """Synthetic frame-0 wrist heights (f0w) and support records (sa) for the self-test."""
    f0w = {
        "a__0": -0.140,
        "a__1": -0.141,
        "a__2": -0.139,
        "b__0": -0.135,
        "b__1": -0.134,
        "a__9": -0.142,
        "c__0": -0.138,
    }
    sa = {
        c: {"desk_mode": "table", "desk_z": -(f0w[c] - (-0.140 if c[0] == "a" else -0.1345)), "hand_zmin": 0.0}
        for c in ("a__0", "a__1", "a__2", "b__0", "b__1")
    }
    sa["a__9"] = {"desk_mode": "table_recovered", "desk_z": -0.02, "hand_zmin": 0.001}
    return f0w, sa


def _selftest_negatives(sa: dict, f0w: dict) -> int:
    """Each check violated on purpose must fail (SystemExit); returns the number of passed cases."""
    ok = 0

    def dies(s, f, mode, allow=False):
        try:
            decide(s, f, mode, allow)
        except SystemExit:
            return True
        return False

    neg = {
        "no mode": (sa, f0w, "kabsch"),
        "|h0| > 1 cm": (sa, dict(f0w, a__9=-0.125), "h1"),
        "policy 0 ↔ measured > 1 cm (h1p oracle = |h0|)": (sa, dict(f0w, a__9=-0.1285), "h1p"),
        "LOO > 6 mm": (dict(sa, a__1=dict(sa["a__1"], desk_z=0.02)), f0w, "h1"),
        "hand below the desk": (dict(sa, a__9=dict(sa["a__9"], hand_zmin=-0.03)), f0w, "h1p"),
        "global fill without explicit flag": (
            dict(sa, c__0={"desk_mode": "table_unknown", "desk_z": -0.01, "hand_zmin": 0.0}),
            f0w,
            "h1",
        ),
        "target missing from the source": (dict(sa, z__0={"desk_mode": "table_unknown", "desk_z": -0.01}), f0w, "h1p"),
    }
    for k, (s, f, m) in neg.items():
        assert dies(s, f, m), f"negative test failed (did not fail): {k}"
        ok += 1
    return ok


def _selftest_idempotent(sa: dict, f0w: dict) -> int:
    """After writing with h1p, rerunning both modes gives the same truth count, targets and values (policy values
    never pose as truth); returns the number of passed modes."""
    ok = 0
    sa2 = json.loads(json.dumps(sa))
    apply_updates(sa2, decide(sa, f0w, WRITE_MODE, False))
    for m in ("h1p", "h1"):
        r1, r2 = decide(sa, f0w, m, False), decide(sa2, f0w, m, False)
        assert (r2["table_n"], r2["targets_n"]) == (r1["table_n"], r1["targets_n"]), (m, r1["table_n"], r2["table_n"])
        assert all(r2["clips"][c]["desk_z"] == r1["clips"][c]["desk_z"] for c in r1["clips"]), m
        ok += 1
    return ok


def _selftest_write(sa: dict, f0w: dict) -> None:
    """Version and write mode, written only into a fake INTAKE in a temporary folder: written clips = decided targets,
    desk_rule{version, h1p} only on those clips, version under recovery, and an h1 write attempt fails before changing
    file bytes."""
    r = decide(sa, f0w, WRITE_MODE, False)
    with tempfile.TemporaryDirectory(prefix="hrdexdb_desk_selftest_") as td:
        ip = Path(td) / "support_audit.json"
        ip.write_text(json.dumps({"clips": sa}))
        assert write_intake(ip, r) == len(r["clips"]) > 0, r["clips"]
        back = json.loads(ip.read_text())
        stamped = {c: v["desk_rule"] for c, v in back["clips"].items() if "desk_rule" in v}
        assert stamped == {c: {"version": RULE_VERSION, "mode": WRITE_MODE} for c in r["clips"]}, stamped
        assert back["recovery"]["hrdexdb_desk_recover"]["rule_version"] == RULE_VERSION, back["recovery"]
        before = ip.read_bytes()
        try:
            write_intake(ip, decide(sa, f0w, "h1", False))
        except SystemExit:
            pass
        else:
            raise AssertionError("negative test failed: --mode h1 wrote INTAKE (no WRITE_MODE guard)")
        assert ip.read_bytes() == before, "negative test failed: an h1 write attempt changed the INTAKE bytes"


def selftest() -> int:
    """Synthetic table: positives (both modes pass · values) + negatives (each check violated on purpose must fail)."""
    f0w, sa = _selftest_table()
    ok = 0
    r = decide(sa, f0w, "h1p", False)
    assert r["clips"]["a__9"]["desk_z"] == 0.0 and abs(r["clips"]["a__9"]["desk_z_h1"] - 0.002) < 1e-9, r
    r = decide(sa, f0w, "h1", False)
    assert abs(r["clips"]["a__9"]["desk_z"] - 0.002) < 1e-9 and r["clips"]["a__9"]["level_src"] == "obj", r
    ok += 2
    ok += _selftest_negatives(sa, f0w)
    ok += _selftest_idempotent(sa, f0w)
    s2 = dict(sa, c__0={"desk_mode": "table_unknown", "desk_z": -0.01, "hand_zmin": 0.0})
    assert decide(s2, f0w, "h1", True)["clips"]["c__0"]["level_src"] == "global_fill"
    ok += 1
    _selftest_write(sa, f0w)
    ok += 2 + _selftest_c2r_direction()
    print(f"[HRDEXDB_DESK_SELFTEST] passed {ok} · failed 0")
    return 0


def _selftest_c2r_direction() -> int:
    """episode_world_poses multiplies by inv(C2R) (official world→robot). An asymmetric C2R (60° · translation)
    separates it from the forward direction; reverting to the forward direction (C2R as is) fails here."""
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    c2r = np.eye(4)
    c2r[:3, :3] = R.from_rotvec([0.0, 0.0, np.pi / 3]).as_matrix()
    c2r[:3, 3] = [0.1, -0.2, 0.5]
    pose = np.eye(4)
    pose[:3, 3] = [0.3, 0.2, 0.6]
    with tempfile.TemporaryDirectory(prefix="hrdexdb_c2r_selftest_") as td:
        (Path(td) / "object_6d").mkdir()
        np.savetxt(Path(td) / "object_6d/pose_000000.txt", pose)
        got = episode_world_poses(Path(td), c2r)[0]
    assert np.allclose(got, np.linalg.inv(c2r) @ pose, atol=1e-12), f"not inv(C2R): {got[:3, 3]}"
    assert not np.allclose(got, c2r @ pose, atol=1e-6), "test premise: C2R does not separate from the forward direction"
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("h1", "h1p"))
    ap.add_argument("--raw", default=str(P.main_root() / "data/hrdexdb/raw/release"))
    ap.add_argument("--out")
    ap.add_argument("--allow-global-level", action="store_true")
    ap.add_argument("--write-intake", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-base", action="store_true")
    ap.add_argument("--hand")
    ap.add_argument("--dataset")
    ap.add_argument("--root", help="data root (the checkout holding the data)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.verify_base:  # verification owner = dexengine.data.support.hrdexdb_verify (the only entry is here)
        from dexengine.data.support import hrdexdb_verify as HV  # noqa: PLC0415

        return HV.verify_main(a, ap)
    if not a.mode or not a.out:
        ap.error("--mode and --out are required (no default)")
    if a.write_intake:
        check_write_mode(a.mode)
    ip = P.layer_path("intake", "hrdexdb") / "support_audit.json"
    res = decide(json.loads(ip.read_text())["clips"], frame0_floors(Path(a.raw)), a.mode, a.allow_global_level)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"hrdexdb_desk_recover_{a.mode}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    d = [abs(r["delta_mm"]) for r in res["clips"].values() if r["delta_mm"] is not None]
    n_fill = sum(r["level_src"] == "global_fill" for r in res["clips"].values())
    print(
        f"[HRDEXDB_DESK] mode={a.mode} · table {res['table_n']} · targets {res['targets_n']} · LOO mm "
        f"{res['loo_mm']} · "
        f"|current−new| max {max(d) if d else 0:.1f} mm · global fill {n_fill}"
    )
    if a.write_intake:
        print(f"[HRDEXDB_DESK] INTAKE updated {write_intake(ip, res)} · desk_rule version {RULE_VERSION} → {ip}")
    else:
        print("[HRDEXDB_DESK] dry — 0 INTAKE writes (only --write-intake writes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
