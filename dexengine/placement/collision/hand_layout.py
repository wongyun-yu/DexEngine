#!/usr/bin/env python3
"""One tree's hand/object layout for the collision sets — qpos addresses, finger joints, abduction joints, main object
bodies and the merged URDF (HandLayout). Read by cross_sets (CrossSetModel) and the cross-repair, gates and audits.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexengine.core import hands as HR
from dexengine.core import qpos_layout

WRIST_DIM = 6  # virtual wrist 6 before the hand block


class HandLayout:
    """One tree's hand/object qpos addresses and finger joint names — one derivation for every hand.

    It only reads its three owners: qpos addresses = `qpos_layout.read(tree)` · finger order and width =
    `core.hands.HAND_REGISTRY` · merged URDF = `full_robot_gen/<arm>_<hand><canon variant>` (sim_collider.combo_urdf;
    the variant's owner is core.arms). Abduction joint = among that finger's joints, the first (most proximal) one whose
    URDF limits are symmetric (lower = −upper) — on SharpA this yields {1,6,10,14,19}·±0.3491, and on wuji it is
    joint2 of finger2~5 (±0.37), with no thumb entry (abduction repair skipped). The same rule applies to a hand seen
    for the first time, and when it cannot be applied that is an exception. Marker: the single `[XR_LAYOUT]` line.
    """

    def __init__(self, tree_dir, urdf=None, quiet: bool = False):
        import json

        tree_dir = Path(tree_dir)
        ti = json.loads((tree_dir / "task_info.json").read_text())
        combo = str(ti.get("combo") or "")
        robot_type = str(ti.get("robot_type") or "")
        if not combo or not robot_type:
            raise ValueError(f"{tree_dir}: task_info has no combo/robot_type — the hand and arm are never guessed")
        from dexengine.core import paths as P  # noqa: PLC0415 — the one combo parser

        try:
            self.arm = P.split_combo(combo, P.ARM_IDS, P.HAND_IDS)[0]
        except ValueError as e:
            raise ValueError(f"{tree_dir}: task_info {e} — the arm is never guessed") from e
        self.spec = _hand_spec_for(robot_type)
        self.hand = self.spec.name
        lay = qpos_layout.read(str(tree_dir))
        self.H = int(lay.hand_qpos_dim)
        self.nq = int(lay.nq)
        self.nF = int(self.spec.n_fingers)
        self.finger_off = self.H - self.nF
        if self.finger_off != WRIST_DIM:
            raise ValueError(
                f"{tree_dir}: hand width {self.H} − fingers {self.nF} = {self.finger_off} ≠ virtual wrist "
                f"{WRIST_DIM} (spec {self.hand} hand_qpos_dim={self.spec.hand_qpos_dim})"
            )
        if self.H != int(self.spec.hand_qpos_dim):  # noqa: SIM300
            raise ValueError(f"{tree_dir}: tree hand width {self.H} ≠ spec {self.hand} {self.spec.hand_qpos_dim}")
        self.hands = {s: int(c) for s, c in lay.hands.items()}  # a hand that is absent has no key
        if sorted(self.hands) != ["left", "right"]:
            raise ValueError(f"{tree_dir}: cross-repair needs a two-hand tree — hands={self.hands}")
        self.obj = {s: (int(lay.main[s]) if s in lay.main else None) for s in ("right", "left")}
        self._object_layout(lay, tree_dir, ti)
        self.finger_names = {"right": tuple(self.spec.finger_joint_names_right), "left": tuple(self.spec.left())}
        self.tips = tuple(self.spec.tip_body_names_right)
        # The finger slot of a link name is computed, never assumed: split the tips on `_`; the column whose values all
        # differ is the finger (the first token only works for wuji · sharpa — DG-5F-S tips are `link_1_4 … link_5_4`).
        self.fkey, self.tokens = HR.finger_column(self.tips, self.hand)
        self.urdf = Path(urdf) if urdf else self._resolve_urdf()
        self.aa = self._aa_joints()  # side → finger idx → (block col, lower, upper) | None
        if not quiet:
            print(
                f"[XR_LAYOUT] arm={self.arm} hand={self.hand} H={self.H} nF={self.nF} finger_off={self.finger_off} "
                f"hands={self.hands} obj={self.obj} obj_body={self.obj_body} shared={self.shared_object} "
                f"obj_vel={self.obj_vel} "
                f"aa={self.aa} urdf={self.urdf}",
                flush=True,
            )

    def _object_layout(self, lay, tree_dir, ti):
        """Object columns, bodies, identities, velocity columns and articulation joints from the tree's qpos layout."""
        # Body name = role=main of objects[] (it can be `extra_object_0` · a shared `right_object`); a tree without
        # such entries uses the `<side>_object` naming rule (explicit, not a fallback).
        self.obj_body: dict[str, str | None] = {"right": None, "left": None}
        mains = [o for o in (lay.objects or []) if o.get("role") == "main"]
        if mains:
            for o in mains:
                for sd in ("right", "left") if o.get("side") == "both" else [o.get("side")]:
                    if sd in self.obj_body:
                        self.obj_body[sd] = o["name"]
        else:
            self.obj_body = {s: (f"{s}_object" if self.obj[s] is not None else None) for s in ("right", "left")}
        for sd in ("right", "left"):
            if (self.obj[sd] is None) != (self.obj_body[sd] is None):
                raise ValueError(
                    f"{tree_dir}: {sd} main object column ({self.obj[sd]}) and body name ({self.obj_body[sd]}) disagree"
                )
        self.shared_object = bool(self.obj_body["right"] and self.obj_body["right"] == self.obj_body["left"])
        from dexengine.core import artifact_paths as tp  # noqa: PLC0415 — owner of the asset id rule

        self.obj_ident = {
            s: (ti.get("dataset_name"), tp.object_asset_id(lay.main_object_id(s), a), bool(a))
            for s in ("right", "left")
            if self.obj[s] is not None
            for a in [lay.main_articulation(s)]
        }  # → objects.hull: (dataset, asset id, articulated)
        # qvel drops 1 column per object, the quaternion's 4th: start column − (number of objects before it)
        starts = sorted({c for c in self.obj.values() if c is not None} | {int(c) for c in lay.extras})
        self.obj_vel = {s: (None if c is None else c - starts.index(c)) for s, c in self.obj.items()}
        # Joint columns (owner qpos_layout.artic) per side and per scene.xml <joint name>; hand_blocks returns
        # an articulated side's object block as 7+1 (+ the joint value), and set_pose runs the part FK with it.
        self.artic = {s: int(a["col"]) for s, a in (lay.artic or {}).items()}
        self.artic_joint = lay.artic_joint_cols()

    # ── The rules ───────────────────────────────────────────────────────────
    def _resolve_urdf(self) -> Path:
        # One resolver for every stage (sim_collider.resolved_urdf): DEXENGINE_XR_URDF, which the stage-2 chain
        # (chain_fast.sh) and record/augment/bake.py also forward to cross_repair as --urdf, else the canon combo URDF.
        from dexengine.placement.collision import sim_collider as SC

        return SC.resolved_urdf(self.arm, self.hand)

    def _aa_joints(self):
        lim = {}
        root = ET.parse(str(self.urdf)).getroot()
        for j in root.findall("joint"):
            lm = j.find("limit")
            if j.get("type") in ("revolute", "continuous") and lm is not None:
                lim[j.get("name")] = (float(lm.get("lower")), float(lm.get("upper")))
        out = {}
        # Joint → finger by the **chain of the merged URDF**: a joint on the ancestor chain of a finger's tip link
        # belongs to that finger. An even split of the declaration order only fits hands with the same joint count per
        # finger (wuji · DG-5F-S 4×5) — XHand Pro has 21 joints (a palm fold `ltspalm` before the little finger chain).
        # On a 4×5 hand the chain rule gives the same assignment and columns as the even split (declarations are
        # finger-major). A joint on two fingers' chains or on no tip's chain dies instead of being guessed.
        _jn = list(self.finger_names["right"])
        up = {j.find("child").get("link"): (j.get("name"), j.find("parent").get("link")) for j in root.findall("joint")}
        owner: dict[str, int] = {}
        for fi, tip in enumerate(self.tips):
            cur, seen = tip, set()
            while cur in up and cur not in seen:
                seen.add(cur)
                jname, cur = up[cur]
                if jname in _jn:
                    if owner.get(jname, fi) != fi:
                        raise ValueError(f"{self.hand}: joint {jname} lies on two finger chains ({owner[jname]}, {fi})")
                    owner[jname] = fi
        lost = [n for n in _jn if n not in owner]
        if lost:
            raise ValueError(f"{self.hand}: finger joints {lost} are on no tip chain ({self.urdf})")
        for fi, tok in enumerate(self.tokens):
            cands = [(k, n) for k, n in enumerate(_jn) if owner[n] == fi]
            if not cands:
                raise ValueError(f"{self.hand}: finger '{tok}' has no joint in the spec")
            sym = [k for k, n in cands if n in lim and lim[n][1] > 0 and abs(lim[n][0] + lim[n][1]) < 1e-6]
            out[fi] = sym[0] if sym else None
        # joint = the symmetric vendor-limit rule · box = per side the reference finger range (the retargeting clip,
        # owner wrist_shift.finger_ranges)
        from dexengine.placement.repair import wrist_shift as _WS  # noqa: PLC0415

        rng = _WS.finger_ranges(self.hand, lim, "bimanual")  # the combo URDF carries both sides
        return {
            sd: {fi: k if k is None else (k, *rng[self.finger_names[sd][k]]) for fi, k in out.items()}
            for sd in ("right", "left")
        }

    # ── Lookups ─────────────────────────────────────────────────────────────
    def hand_blocks(self, qpos_row):
        """One frame of tree qpos → (finger block right/left, object 7 right/left (None when absent))."""
        f = {s: qpos_row[c + self.finger_off : c + self.H] for s, c in self.hands.items()}
        o = {s: (None if c is None else qpos_row[c : c + 7]) for s, c in self.obj.items()}
        for s, col in self.artic.items():  # artic: pose 7 + joint 1 (others stay 7)
            if o.get(s) is not None:
                o[s] = np.concatenate([np.asarray(o[s], float), [float(qpos_row[col])]])
        return f, o

    def finger_of(self, link: str):
        """Link → finger index (thumb 0 … little finger 4), or None."""
        parts = HR.strip_side(link).split("_")
        if self.fkey >= len(parts):
            return None
        try:
            return self.tokens.index(parts[self.fkey])
        except ValueError:
            return None


def _hand_spec_for(robot_type: str):
    """Tree `robot_type` (a retargeting directory name or a hand name) → HandSpec. Exactly one must match."""
    hits = {
        sp.name: sp
        for sp in HR.HAND_REGISTRY.values()
        if sp is not None and robot_type in (sp.name, getattr(sp, "retarget_dirname", ""))
    }
    if len(hits) != 1:
        raise ValueError(
            f"robot_type={robot_type!r} matches {len(hits)} HandSpecs: {sorted(hits)} — register exactly one"
        )
    return next(iter(hits.values()))
