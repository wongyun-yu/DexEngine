"""Single owner of the clip phase spans (approach, grasp, manipulate, release, retreat): one rule, segment, and two
signals.

Rule (segment): inputs = per-step hand-object contact (bool) · object positions [T, k, 3] (· orientations wxyz
  [T, k, 4]) · dt.
  contact span [c0, c1] = first and last steps with contact (contact blobs shorter than MIN_ON_S are dropped).
  motion span [m0, m1] = the first step where any object left its base place (time BASE_S; the settling motion right
  after the start is excluded) by more than MOVE_M (or ROT_DEG) · the last step it was that far from its end place.
  approach = [0, c0) · grasp = [c0, m0) · manipulate = [m0, m1] · release = (m1, c1] · retreat = (c1, T); while an
  object moves it counts as in contact (widened to c0 <= m0, c1 >= m1: steps the contact signal missed).
  No contact: contact span = motion span · no motion: no manipulate (grasp = contact span: pressing, touching) ·
  neither: approach alone. Starting in contact (c0 <= BASE_S: a piece that starts already held, or the force caught a
  few steps late) -> no grasp, manipulate from 0 · ending in contact (c1 >= end - BASE_S: a piece that ends held) -> no
  release, manipulate to the end (the grasp or release moment is not in the clip).
  Phases shorter than MIN_PHASE_S merge into the previous phase (the next one for the first) · no empty phases (piece
  clips have only some phases). Returns [{phase, i0, i1}] (i1 included), covering [0, T - 1] in order without gaps.
Signals (same rule, different time axes; frames and the phase table must share a time axis):
  from_episode   a record episode (policy steps; body from_files, also during the store): contact = link max
                 ‖F‖ > CONTACT_N of force_partner.<side>_hand_contact.object_*.force_n_clean (union over hands and
                 objects; force time -> nearest step) · objects = observation.object_pose.* columns.
  from_reference the source reference motion (trajectory_kinematic rows; used only for text requests of clips without
                 a render): contact = contact_mano_<active side>.npz contact (fingers, any region) ∨ palm
                 effective_dist < contact_threshold_m · objects = qpos[:, qpos_start:+7].
frame_picks: up to n times in order phase middles -> boundaries (GRASP, MANIPULATE start, MANIPULATE end, RETREAT
  start); times within FRAME_GAP_S of a chosen one are dropped (the same picture twice).
text_picks: text request frame plan (TEXT_FRAMES: view -> count); exo = exo_picks (phase middles -> boundaries ->
  manipulate ¼·¾ -> first and last -> even fill; needs the video length) · other views = frame_picks. A view missing
  from the plan dies.
A read-only module (numpy and pyarrow inside functions); the render session (render.framing) reads MOVE_M and BASE_S
from here too.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.store.parquet import read_table  # threads off (avoids exit rc 134)

PHASES = ("approach", "grasp", "manipulate", "release", "retreat")
MOVE_M = 0.01  # an object moved = 1 cm from its base place
ROT_DEG = 10.0  # or turned 10°
BASE_S = 0.4  # base place = this time (first pose at rest; excludes the settling right after the start)
CONTACT_N = 0.5  # hand-object contact = any link above this force (N) (after the store.force quality cleanup)
MIN_ON_S = 0.1  # contact blobs shorter than this are dropped as noise
MIN_PHASE_S = 0.3  # phases shorter than this merge into a neighbour
FRAME_CAP = 8  # cap of frame times per view
FRAME_GAP_S = 0.25  # minimum gap between frame times
TEXT_FRAMES = {"exo": 8, "egoL": 4}  # text request frame plan (view -> count; total 12)


def _runs(mask) -> list[tuple[int, int]]:
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def contact_span(contact, dt: float) -> tuple[int, int] | None:
    keep = [r for r in _runs(list(map(bool, contact))) if (r[1] - r[0] + 1) * dt >= MIN_ON_S]
    return (keep[0][0], keep[-1][1]) if keep else None


def _ang(q0, q1) -> float:
    d = min(1.0, abs(sum(a * b for a, b in zip(q0, q1, strict=True))))
    return math.degrees(2 * math.acos(d))


def moving_span(pos, quat, dt: float) -> tuple[int, int] | None:
    """pos [T, k, 3] · quat [T, k, 4] | None -> (m0, m1) | None (no motion)."""
    import numpy as np

    P = np.asarray(pos, float).reshape(len(pos), -1, 3)
    T = len(P)
    if T == 0 or P.shape[1] == 0:
        return None
    Q = None if quat is None else np.asarray(quat, float).reshape(T, -1, 4)
    b = min(T - 1, int(round(BASE_S / dt)))

    def off(ref: int):
        d = np.linalg.norm(P - P[ref], axis=2).max(1) > MOVE_M
        if Q is not None:
            d |= np.array([max(_ang(Q[t, j], Q[ref, j]) for j in range(Q.shape[1])) > ROT_DEG for t in range(T)])
        return d

    a = off(b)
    a[:b] = False
    if not a.any():
        return None
    m0 = int(np.flatnonzero(a)[0])
    e = np.flatnonzero(off(T - 1))
    m1 = max(m0, int(e[-1])) if len(e) else m0
    return m0, m1


def segment(contact, pos, quat, dt: float) -> list[dict]:
    """Rule (module doc) -> [{phase, i0, i1}]."""
    T = len(contact)
    if T == 0 or len(pos) != T:
        raise ValueError(f"[PHASES] lengths differ: contact {T} · objects {len(pos)}")
    c, m = contact_span(contact, dt), moving_span(pos, quat, dt)
    if c is None:
        c = m
    if c is None:
        return [{"phase": "approach", "i0": 0, "i1": T - 1}]
    c0, c1 = c
    if m is None:
        cuts = [("approach", 0, c0 - 1), ("grasp", c0, c1), ("retreat", c1 + 1, T - 1)]
    else:
        m0, m1 = m
        c0, c1 = min(c0, m0), max(c1, m1)  # in contact while the object moves (even when the contact signal missed)
        edge = int(round(BASE_S / dt))  # start and end settling window (force and motion caught late)
        if c0 <= edge:  # starts held: the grasp moment is not in the clip
            c0 = m0 = 0
        if c1 >= T - 1 - edge:  # ends held: the release moment is not in the clip
            c1 = m1 = T - 1
        cuts = [
            ("approach", 0, c0 - 1),
            ("grasp", c0, m0 - 1),
            ("manipulate", m0, m1),
            ("release", m1 + 1, c1),
            ("retreat", c1 + 1, T - 1),
        ]
    ph = [{"phase": p, "i0": i0, "i1": i1} for p, i0, i1 in cuts if i1 >= i0]
    return _merge_short(ph, dt)


def _merge_short(ph: list[dict], dt: float) -> list[dict]:
    """Merge phases shorter than MIN_PHASE_S into the previous one (the next for the first), until one phase is left."""
    ph = [dict(p) for p in ph]
    while len(ph) > 1:
        k = next((i for i, p in enumerate(ph) if (p["i1"] - p["i0"] + 1) * dt < MIN_PHASE_S), None)
        if k is None:
            break
        if k == 0:
            ph[1]["i0"] = ph[0]["i0"]
        else:
            ph[k - 1]["i1"] = ph[k]["i1"]
        ph.pop(k)
    return ph


def table(ph: list[dict], dt: float) -> list[dict]:
    """Phase table for requests and records: [{phase, start_s, end_s}]."""
    return [{"phase": p["phase"], "start_s": round(p["i0"] * dt, 2), "end_s": round((p["i1"] + 1) * dt, 2)} for p in ph]


def from_meta(ds: Path, e: dict) -> tuple[list[dict], float]:
    """The loaded episode's meta row subtasks (phases measured by from_files at store and reload, policy steps) ->
    (phases, dt = 1/fps); readable from copies without the body (parquet). A missing subtasks field dies."""
    if "subtasks" not in e:
        raise InputMissingError(f"[PHASES] {ds} ep{e.get('episode_index')}: no subtasks field in the meta")
    fps = float(json.loads((Path(ds) / "meta/info.json").read_text())["fps"])
    return [
        {"phase": s["phase"], "i0": int(s["start_step"]), "i1": int(s["end_step"])} for s in e["subtasks"]
    ], 1.0 / fps


def full_table(rows: list[dict]) -> list[dict]:
    """Text request phase table (v3): always carries the five PHASES; existing rows as they are, missing phases as
    other_replay (no times: not in this replay, but possible in another replay, hand or variant of the same clip).
    Since the text has all five phases, the phase text of store and reload (store.tasks.episode_subtasks) is never
    empty for any episode: the structural fix of empty phase texts (the same phase set at build and load = the one
    PHASES). Unknown phases die."""
    have = {r["phase"]: r for r in rows}
    if bad := set(have) - set(PHASES):
        raise SystemExit(f"[PHASES] unknown phases {sorted(bad)}: PHASES {PHASES}")
    return [have.get(p) or {"phase": p, "other_replay": True} for p in PHASES]


def union_table(ph: list[dict], dt: float, others: list[tuple[list[dict], float]]) -> list[dict]:
    """Text input phase table: the frame episode's phase table (table) + phases only other loaded episodes of the same
    clip have (other hands, rows) (other_replay=True, no times: another episode's clock is not placed on the frame
    episode's time axis). Order = PHASES · unknown phases die."""
    rows = {r["phase"]: r for r in table(ph, dt)}
    for oph, odt in others:
        for r in table(oph, odt):
            rows.setdefault(r["phase"], {"phase": r["phase"], "other_replay": True})
    if bad := set(rows) - set(PHASES):
        raise SystemExit(f"[PHASES] unknown phases {sorted(bad)}: PHASES {PHASES}")
    return [rows[p] for p in PHASES if p in rows]


def frame_picks(ph: list[dict], dt: float, n: int = FRAME_CAP) -> list[dict]:
    """Frame times: phase middles first, then boundaries (grasp, manipulate start, manipulate end, retreat start), up
    to n in time order.
    [{t_s, phase, at: mid|start|end}]."""
    picks = [{"t_s": round((p["i0"] + p["i1"] + 1) / 2 * dt, 2), "phase": p["phase"], "at": "mid"} for p in ph]
    by = {p["phase"]: p for p in ph}
    for name, at in (("grasp", "start"), ("manipulate", "start"), ("manipulate", "end"), ("retreat", "start")):
        if name in by:
            i = by[name]["i0"] if at == "start" else by[name]["i1"]
            picks.append({"t_s": round(i * dt, 2), "phase": name, "at": at})
    out = []
    for p in picks:
        if all(abs(p["t_s"] - q["t_s"]) >= FRAME_GAP_S for q in out):
            out.append(p)
    return sorted(out[:n], key=lambda p: p["t_s"])


def exo_picks(ph: list[dict], dt: float, dur: float, n: int) -> list[dict]:
    """Wide view (exo) frame times: phase middles -> boundaries (grasp, manipulate start, manipulate end, release,
    retreat start) -> manipulate ¼ and ¾ -> first and last (dur - 0.05) -> evenly (dur·k/9), up to n in order; times
    within FRAME_GAP_S of a chosen one are dropped. Times are measured on the phase table (end = (i1 + 1)·dt).
    [{t_s, phase, at}] in time order."""
    tab = table(ph, dt)
    by = {p["phase"]: p for p in tab}

    def phase_at(t):
        return next((p["phase"] for p in tab if p["start_s"] <= t <= p["end_s"]), tab[-1]["phase"] if tab else "")

    cand = [(round((p["start_s"] + p["end_s"]) / 2, 2), p["phase"], "mid") for p in tab]
    for name, at in (
        ("grasp", "start"),
        ("manipulate", "start"),
        ("manipulate", "end"),
        ("release", "start"),
        ("retreat", "start"),
    ):
        if name in by:
            cand.append((by[name]["start_s"] if at == "start" else by[name]["end_s"], name, at))
    if "manipulate" in by:
        m = by["manipulate"]
        cand += [
            (round(m["start_s"] + q * (m["end_s"] - m["start_s"]), 2), "manipulate", lab)
            for q, lab in ((0.25, "q1"), (0.75, "q3"))
        ]
    cand += [(0.0, phase_at(0.0), "first"), (round(dur - 0.05, 2), phase_at(dur - 0.05), "last")]
    cand += [(round(dur * k / 9, 2), phase_at(dur * k / 9), "fill") for k in range(1, 9)]
    out = []
    for t, p, at in cand:
        if len(out) < n and all(abs(t - q[0]) >= FRAME_GAP_S for q in out):
            out.append((t, p, at))
    return [{"t_s": t, "phase": p, "at": at} for t, p, at in sorted(out)]


def text_picks(ph: list[dict], dt: float, view: str, dur: float) -> list[dict]:
    """Text request frame times (one view): the TEXT_FRAMES count · exo = exo_picks · others = frame_picks. A view
    missing from the plan dies."""
    if view not in TEXT_FRAMES:
        raise InputMissingError(
            f"[PHASES] view {view!r} is not in the text frame plan TEXT_FRAMES {sorted(TEXT_FRAMES)}"
        )
    n = TEXT_FRAMES[view]
    return exo_picks(ph, dt, dur, n) if view == "exo" else frame_picks(ph, dt, n)


# ───────────────────────── signals ─────────────────────────
def from_episode(ds: Path, i: int) -> tuple[list[dict], float]:
    """Stored dataset episode i (policy steps) -> (phases, dt); places from the path templates of meta/info.json."""
    info = json.loads((Path(ds) / "meta/info.json").read_text())
    ch = i // int(info.get("chunks_size", 1000))
    return from_files(
        Path(ds) / info["data_path"].format(episode_chunk=ch, episode_index=i),
        Path(ds) / info["collect"]["force_path"].format(episode_chunk=ch, episode_index=i),
    )


def from_files(parquet: Path, forces: Path) -> tuple[list[dict], float]:
    """Episode table (parquet) and forces (npz) -> (phases, dt); usable during the store too (before info.json).
    Missing force or object columns die."""
    import numpy as np

    ds, i = parquet.parent, parquet.stem
    t = read_table(parquet).to_pydict()
    # absent objects (shape NaN fields) are dropped
    keys = sorted(
        k for k in t if k.startswith("observation.object_pose.") and not np.isnan(np.asarray(t[k], float)).all()
    )
    if not keys:
        raise InputMissingError(f"[PHASES] {ds} ep{i}: no object pose columns (observation.object_pose.*)")
    X = np.stack([np.asarray(t[k], float) for k in keys], axis=1)  # [T, k, 7]
    ts = np.asarray(t["timestamp"], float)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.02
    z = np.load(forces, allow_pickle=False)
    fk = [k for k in z.files if k.startswith("force_partner.") and ".object_" in k and k.endswith(".force_n_clean")]
    if not fk:
        raise InputMissingError(f"[PHASES] {ds} ep{i}: no hand-object forces (force_partner.*.object_*.force_n_clean)")
    F = np.max([np.linalg.norm(z[k], axis=2).max(1) for k in fk], axis=0)  # [n_force]
    ft = np.asarray(z["t"], float)
    j = np.clip(np.searchsorted(ft, ts), 0, len(ft) - 1)
    j = np.where((j > 0) & (np.abs(ft[np.maximum(j - 1, 0)] - ts) < np.abs(ft[j] - ts)), j - 1, j)
    return segment(F[j] > CONTACT_N, X[:, :, :3], X[:, :, 3:7], dt), dt


def human_touch(tree: Path, side: str):
    """Frames [T] bool where that hand touched its object in the human data (source reference motion): contact (fingers,
    any region) of the contact sidecar (GT) tree/0/contact_mano_<side>.npz ∨ palm effective_dist < contact_threshold_m.
    Readers (no copy): from_reference (phase signal) · rollout.recorder oracle_gate (the record oracle's '0 touched
    samples' gate). A missing file dies (unknown is never read as 'not touched')."""
    import numpy as np

    f = Path(tree) / "0" / f"contact_mano_{side}.npz"
    if not f.is_file():
        raise InputMissingError(f"[PHASES] {tree}: {f.name} missing")
    c = np.load(f, allow_pickle=False)
    return c["contact"].reshape(len(c["contact"]), -1).any(1) | (c["palm_effective_dist"] < c["contact_threshold_m"])


def from_reference(tree: Path) -> tuple[list[dict], float]:
    """Source reference motion (clip tree 0/trajectory_kinematic.npz rows) -> (phases, dt)."""
    import numpy as np

    tree = Path(tree)
    ti = json.loads((tree / "task_info.json").read_text())
    kz = np.load(tree / "0" / "trajectory_kinematic.npz", allow_pickle=False)
    q = kz["qpos"]
    dt = 1.0 / float(kz["frequency"]) if "frequency" in kz.files else float(ti["ref_dt"])
    objs = [o for o in ti.get("objects") or [] if o.get("qpos_start") is not None]
    if not objs:
        raise InputMissingError(f"[PHASES] {tree}: no object qpos_start")
    X = np.stack([q[:, o["qpos_start"] : o["qpos_start"] + 7] for o in objs], axis=1)
    on = np.zeros(len(q), bool)
    sides = ti.get("active_sides") or []
    if not sides:
        raise InputMissingError(f"[PHASES] {tree}: no active_sides")
    for s in sides:
        hit = human_touch(tree, s)
        if len(hit) != len(q):
            raise SystemExit(f"[PHASES] {tree}: contact_mano_{s}.npz rows {len(hit)} != kinematic rows {len(q)}")
        on |= hit
    return segment(on, X[:, :, :3], X[:, :, 3:7], dt), dt
