#!/usr/bin/env python3
"""Robot self-penetration of a delivered trajectory: the owner of rule J-self-pen (threshold, counted pair classes,
measurement, verdict, merge reason and the manifest statement).

Rule: a delivered trajectory whose executed joints (`observation.joint_pos.robot`, every frame) have the robot inside
itself by more than ROBOT_SELF_PEN_MAX_M on a counted pair class is dropped. The unit is the trajectory; a clip leaves
only when every trajectory of it is dropped. Measured on the output, since training runs with robot self-collision off
and a clean reference can still give a penetrating rollout.
Joints read: every movable joint of the merged URDF, by name from the recorded table — arms, head and both hands'
fingers (HandSpec order). No neutral fill: a movable joint the table does not name, or a recorded joint that is not
movable in the URDF (a value the ruler could not place), raises. The torso is fixed joints of the canonical-variant URDF
(the body the sim spawns), so it has no recorded value. Ruler = cross_sets.CrossSetModel (the canon collision sets; no
second ruler): sets
RULER_SETS, robot geometry RULER_GEOM (arm and torso links = the sim colliders, hand links = the URDF <collision>
bodies), the URDF of sim_collider.resolved_urdf, no objects (the two sets read none). Depth = minus the signed distance
of mj_geomDistance (GJK/EPA).
Pair classes = the ruler's class strings mapped by CLASS_OF (link_shapes.group_of puts the EE link with the hand):
  AA arm-arm · AT arm <-> torso · HT hand <-> torso · HA hand <-> the other arm · HH hand-hand.
COUNTED = AA AT HT HA; HH is not counted. A class string outside CLASS_OF raises: a new ruler class is never silently
left uncounted.

Layers: worst (ruler rows of one frame -> the deepest counted pair) · measure (joint arrays, every frame) · verdict ·
marker · reason · statement · measure_joints (a recorded joint table by name) · measure_parquet (one episode file).
Wired in the final merge (record.store.merge_self_pen): a dropped trajectory's episodes are not merged and the pieces
line records them.
Marker: `[SELF_PEN] <trajectory> max <mm> pair <a>|<b> frame <f> -> keep|drop`, one line per measured trajectory.

  <retarget venv> -m dexengine.placement.collision.self_pen <episode parquet> --info <meta/info.json>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Drop line: a counted pair deeper than this (m) in any frame of a recorded trajectory drops that trajectory.
ROBOT_SELF_PEN_MAX_M = 0.020
#: The canon collision sets measured: crossed hand/arm <-> hand/arm and arm/hand <-> torso.
RULER_SETS = ("1_cross", "4_torso")
#: Robot geometry of the ruler (cross_sets / link_shapes.ROBOT_GEOMS).
RULER_GEOM = "collision"
#: Query distance of the ruler (m): rows nearer than this come back; only penetrations (< 0) are counted.
QUERY_DISTMAX_M = 0.01
#: Ruler class string -> pair class.
CLASS_OF = {
    "arm-arm": "AA",
    "Rarm-torso": "AT",
    "Larm-torso": "AT",
    "Rhand-torso": "HT",
    "Lhand-torso": "HT",
    "hand-arm": "HA",
    "arm-hand": "HA",
    "hand-hand": "HH",
}
#: Pair classes the rule counts (hand-hand is not counted).
COUNTED = ("AA", "AT", "HT", "HA")
#: The recorded column measured (LeRobot feature of the delivered episodes).
JOINTS = "observation.joint_pos.robot"
MARK = "[SELF_PEN]"
REASON_HEAD = "self_pen"
STATEMENT_KEYS = ("class", "frame", "frames", "max_mm", "pair", "rule", "state", "verdict")
SIDES = ("right", "left")


@dataclass(frozen=True)
class Result:
    """The deepest counted penetration of one trajectory: depth (m, 0.0 = none), its class and links ('' = none), its
    frame (-1 = none) and the number of frames measured."""

    depth_m: float
    cls: str
    a: str
    b: str
    frame: int
    frames: int


def rule() -> dict:
    """The rule part of a statement: threshold, counted classes and the ruler."""
    return {
        "threshold_mm": round(ROBOT_SELF_PEN_MAX_M * 1000.0, 3),
        "counted": list(COUNTED),
        "ruler": {"module": "dexengine.placement.collision.cross_sets", "sets": list(RULER_SETS), "geom": RULER_GEOM},
        "measured": JOINTS,
    }


def worst(rows) -> tuple[float, str, str, str]:
    """Ruler rows of one frame [(set, class, link a, link b, distance, fromto)] -> (depth m, class, a, b) of the deepest
    counted penetration, (0.0, '', '', '') when there is none. An unknown class raises."""
    best = (0.0, "", "", "")
    for _set, cls, a, b, dist, _ft in rows:
        if cls not in CLASS_OF:
            raise ValueError(f"{MARK} ruler class {cls!r} has no pair class (known {sorted(CLASS_OF)})")
        if CLASS_OF[cls] in COUNTED and -float(dist) > best[0]:
            best = (-float(dist), CLASS_OF[cls], a, b)
    return best


def measure(model, body: dict, fingers: dict, base) -> Result:
    """Every frame of a trajectory on the ruler model: body {side: (joint names, (T, k))} = the non-finger joints, set
    by name (CrossSetModel.build_q) · fingers {side: (T, nF)} · base = the robot base position (the robot-only sets do
    not depend on it). The four arrays must have one length."""
    lens = {f"body {s}": len(body[s][1]) for s in SIDES} | {f"fingers {s}": len(fingers[s]) for s in SIDES}
    if len(set(lens.values())) != 1:
        raise ValueError(f"{MARK} joint arrays of different lengths {lens}")
    T = lens["body right"]
    best, at = (0.0, "", "", ""), -1
    for t in range(T):
        model.set_pose({s: (body[s][0], body[s][1][t]) for s in SIDES}, {s: fingers[s][t] for s in SIDES}, {}, base)
        w = worst(model.distances(QUERY_DISTMAX_M, sets=RULER_SETS)[2])
        if w[0] > best[0]:
            best, at = w, t
    return Result(best[0], best[1], best[2], best[3], at, T)


def verdict(res: Result) -> str:
    return "drop" if res.depth_m > ROBOT_SELF_PEN_MAX_M else "keep"


def marker(clip: str, res: Result) -> str:
    """The [SELF_PEN] line (mm with two decimals, as the statement: 20.004 mm never reads "20.0 -> drop")."""
    return (
        f"{MARK} {clip} max {res.depth_m * 1000.0:.2f} pair {res.a or '-'}|{res.b or '-'} frame {res.frame} -> "
        f"{verdict(res)}"
    )


def reason(res: Result) -> str | None:
    """The drop reason (head REASON_HEAD), None for a keep."""
    if verdict(res) == "keep":
        return None
    return f"{REASON_HEAD}{res.depth_m * 1000.0:.2f}mm({res.a}|{res.b}@{res.frame})"


def statement(res: Result) -> dict:
    """The manifest statement of one measured trajectory (keys STATEMENT_KEYS)."""
    return {
        "state": "measured",
        "max_mm": round(res.depth_m * 1000.0, 2),
        "class": res.cls,
        "pair": f"{res.a}|{res.b}" if res.a else "",
        "frame": res.frame,
        "frames": res.frames,
        "verdict": verdict(res),
        "rule": rule(),
    }


_MODELS: dict = {}


def model_for(layout):
    """The ruler model of a layout's URDF and fingers (no objects), built once per process."""
    from dexengine.placement.collision import cross_sets as CC  # noqa: PLC0415

    key = (str(layout.urdf), tuple(layout.finger_names["right"]), tuple(layout.finger_names["left"]))
    if key not in _MODELS:
        _MODELS[key] = CC.CrossSetModel(layout, tree_dir=None, geom=RULER_GEOM)
    return _MODELS[key]


class RobotLayout:
    """What the ruler reads of the robot (CrossSetModel's layout): the merged URDF of (arm, hand), the finger joint
    names per side (HandSpec order) and no object bodies."""

    def __init__(self, arm: str, hand: str):
        from dexengine.placement.collision import sim_collider as SC  # noqa: PLC0415
        from dexengine.placement.collision.hand_layout import _hand_spec_for  # noqa: PLC0415

        spec = _hand_spec_for(hand)
        self.arm, self.hand = arm, spec.name
        self.finger_names = {"right": tuple(spec.finger_joint_names_right), "left": tuple(spec.left())}
        self.urdf = SC.resolved_urdf(arm, spec.name)
        self.obj_body = {"right": None, "left": None}


def body_joints(model, layout) -> list:
    """The movable joints of the ruler's URDF that are not fingers (arms, head), in model order."""
    fing = {n for s in SIDES for n in layout.finger_names[s]}
    return [n for n in model.jq if n not in fing]


def measure_joints(combo: str, names: list, Q) -> Result:
    """A recorded robot joint table Q (T, len(names)) of the robot `combo` (`<arm>_<hand>`, the dataset's robot_type)
    on the ruler, every movable joint set from its column (module doc: no neutral fill)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    arm, hand, _variant = P.split_combo(combo, P.ARM_IDS, P.HAND_IDS)
    layout = RobotLayout(arm, hand)
    model = model_for(layout)
    body = body_joints(model, layout)
    ix = {n: i for i, n in enumerate(names)}
    gone = [n for n in [*body, *layout.finger_names["right"], *layout.finger_names["left"]] if n not in ix]
    extra = [n for n in names if n not in model.jq]
    if gone or extra:
        raise KeyError(
            f"{MARK} recorded joints vs the movable joints of {Path(layout.urdf).name}: missing {gone[:4]} · not "
            f"movable {extra[:4]} — no neutral fill"
        )
    Q = np.asarray(Q, float)
    body_q = {"right": (body, Q[:, [ix[n] for n in body]]), "left": ([], Q[:, []])}
    fingers = {s: Q[:, [ix[n] for n in layout.finger_names[s]]] for s in SIDES}
    return measure(model, body_q, fingers, np.zeros(3))


def read_joints(parquet) -> np.ndarray:
    """The executed robot joints (T, n) of one episode file."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    col = pq.read_table(str(parquet), columns=[JOINTS]).column(0).to_pylist()
    return np.asarray(col, float)


def table_key(Q) -> str:
    """The identity of one joint table (shape and bytes): the looks of one rollout share it."""
    import hashlib  # noqa: PLC0415

    Q = np.ascontiguousarray(Q)
    return hashlib.sha256(repr((Q.shape, Q.dtype.str)).encode() + Q.tobytes()).hexdigest()


def measure_parquet(parquet, info: dict) -> Result:
    """One episode file of a LeRobot dataset whose meta/info.json is `info` (robot_type + joint names)."""
    names = list(info["features"][JOINTS]["names"])
    return measure_joints(str(info["robot_type"]), names, read_joints(parquet))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parquet")
    ap.add_argument("--info", required=True, help="the dataset's meta/info.json")
    a = ap.parse_args(argv)
    res = measure_parquet(a.parquet, json.loads(Path(a.info).read_text(encoding="utf-8")))
    print(marker(Path(a.parquet).name, res), flush=True)
    print(json.dumps(statement(res), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
