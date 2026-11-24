"""Rule thresholds that may depend on hand geometry, and the frame each one is measured in."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from dexengine.core.rules.identity import active_arm, active_hand
from dexengine.core.rules.rule import RULES
from dexengine.core.rules.ruleset import RULESET_VERSION
from dexengine.core.rules.source import _sym

#: **Thresholds tied to hand geometry** — the one owner of what silently changes when the hand changes.
#:
#: Whether a threshold depends on the hand is decided not by its **value** but by the **frame** it is measured in
#: (no hand is the base of any other hand). D35 takes the contact, wrist and fingertip-speed source from the **human
#: hand (MANO)**, so a threshold measured in the MANO frame gives the same answer whatever the hand.
#:
#: So this table does not record "which hand it was derived from". It reads, from the owner module, **which frame
#: each threshold's reading path measures in right now**, and that one frame decides hand dependence (FRAME_* below).
#: It is a rule, not a lookup table — it applies unchanged to a hand never seen before, and if the source goes back
#: to `robot` the same rule says "hand-dependent" again.
#:
#: The values are not derived. 0.30 · −0.05 · 0.03 · 0.01 are all **chosen policy numbers**; no formula
#: in HandSpec (fingertip pad offsets · palm link · mount offset · hand length) produces these four numbers. All are
#: round multiples of 0.01 m. A formula that pretends to produce them from hand geometry would be **a fit to these four
#: numbers**, not a derivation, so none is written. Hand independence comes from the **frame**, not the value.

#: The frame a threshold is measured in. **The name is the rule** — hand dependence is decided by this alone.
FRAME_MANO = "mano"  #: measured on human-hand (MANO) keypoints → the same answer for every robot hand
FRAME_ROBOT = "robot"  #: measured in the retargeted robot-hand frame → the answer changes with the hand
FRAME_RATIO = "ratio"  #: dimensionless ratio → no frame to measure in (a time / frame ratio, not a length)
FRAME_NONE = "no-hand"  #: the hand is not in the formula (object–object distances etc.)


def case_source() -> str:
    """What cases, roles and wrist height are measured from **right now** — read from the one owner of D35.

    No value is written here. `contact_source` is a module with no dependencies, so it is imported directly — that is
    why the module exists (no copies).
    """
    from dexengine.data.contact.source import CONTACT_SOURCE  # noqa: PLC0415

    return str(CONTACT_SOURCE)


def _frame_case() -> str:
    """Quantities measured in the case layer — human hand if the source is mano, robot hand if robot."""
    return FRAME_MANO if case_source() == "mano" else FRAME_ROBOT


def _frame_mano_always() -> str:
    """A path that measures on human-hand keypoints regardless of the source (e.g. oakink2 raw object contact
    detection)."""
    return FRAME_MANO


def _frame_ratio() -> str:
    return FRAME_RATIO


def _frame_none() -> str:
    return FRAME_NONE


@dataclass(frozen=True)
class HandThreshold:
    """One threshold and **the paths that read it**. The value is read from the owner file (no copy here)."""

    key: str
    """Short name used in the log marker (`[THRESH] … D18=…`)."""

    rule: str
    """The D rule id(s) that use this threshold (several are joined)."""

    owner: str
    """The one owner of the value, `module:symbol`. `_sym` reads it **from that file**."""

    consumers: tuple
    """((consumer path name, frame resolver), …). The resolver **computes** the frame at run time."""

    why: str
    """What ties this threshold to its frame — only statements that can be checked in code."""

    waiver: str = ""
    """Why a hand-dependent path that remains is not fixed now (a tracked reason is expected). The report below
    prints it; no check stops a run on an empty waiver."""


HAND_THRESHOLDS = (
    HandThreshold(
        key="D14'",
        rule="D14'·D14-fb",
        owner="dexengine/data/candidate_rules.py:WRIST_BELOW_M",
        consumers=(
            ("case.wrist_fp(grab_desk_oracle.wrist_footprint)", _frame_mano_always),
            ("case.wrist_desk_m(D14-fb)", _frame_case),
        ),
        why="Two paths read the same −0.05 m, and **both measure on the human hand (MANO)** (D39). "
        "On the grab canonical path the case stage feeds, **per tree window**, the wrist of "
        "`mano_frame.mano_keypoints_base` (**the same function** as the case `wrist_desk_m` the fallback D14-fb "
        "reads) into `grab_desk_oracle.wrist_footprint` and measures the lowest wrist point inside the footprint "
        "(without a MANO source it gives no value and candidate selection dies — there is no robot-frame path). "
        "D39 also moved the oracle to MANO, so no robot-frame reading of this value is left.",
    ),
    HandThreshold(
        key="D28-palm",
        rule="D28",
        owner="dexengine/data/cases/hand_object_assign.py:TH_PALM_M",
        consumers=(("case.palm_dist", _frame_case),),
        why="Palm reference point ↔ object surface. With the mano source that point is a **human-hand proxy** "
        "`wrist + α·(mean fingertip − wrist)` (contact_sidecar.compute_side, α=contact_sidecar.palm_alpha). "
        "α=0.4 was chosen **with the wuji robot palm as the oracle** (the only hand with collision geometry on its "
        "palm link, so it became the baseline), but the point α places lies **on the MANO skeleton**. "
        "Oracle and measured object differ — the skeleton is the same for every hand, so is the answer. So it "
        "is not recalibrated for dg5f_s (that would create a hand dependence that is not there).",
    ),
    HandThreshold(
        key="D28-pad",
        rule="D28",
        owner="dexengine/data/contact/common.py:KEYPOINT_PAD_M",
        consumers=(("case.tip contact(TH_CONTACT_M)", _frame_case), ("case.objobj(object–object)", _frame_none)),
        why="Skin thickness added to the keypoint (joint centre) ↔ surface distance. With the mano source the "
        "fingertip threshold applies to the **MANO fingertip keypoints**, and the applied value comes from the "
        "measured file (case_taxonomy `_MANO_THR` ← configs/contact/contact_source_distance.json:best_thr_m = "
        "0.0275; only without the file does it fall back to the sidecar's 0.02 = 0.01+PAD). That 0.0275 is also "
        "a **human-hand operating point** calibrated against wuji robot contact, so it is the same for every "
        "hand (the robot-source branch that put this pad on robot fingertip surface points is gone).",
    ),
    HandThreshold(
        key="D16",
        rule="D16",
        owner="dexengine/data/cases/hand_object_assign.py:TH_ACTIVE_MIN_S",
        consumers=(("case.active(summed grasp time)", _frame_case),),
        why="An active hand = summed grasp time (s) ≥ one threshold (the contact ratio is not an activity criterion — "
        "only the case's informational mark touch_only uses ratios). Grasp comes from the fingertip / palm "
        "distances of the contact source (D35), so with the mano source it is in the human-hand frame and "
        "**hand-free**.",
    ),
    HandThreshold(
        key="D26/D24′",
        rule="D26′·D24′",
        owner="dexengine/placement/coords/thresholds.py:HELD_CONTACT_FRAC",
        consumers=(("coords._held_in_interval", _frame_ratio),),
        why="The contact-frame ratio (0.5) above which an object counts as 'held in hand' during a still interval. "
        "Dimensionless for the same reason as D16, and the sidecar that makes the ratio is MANO, so there is no "
        "place for the hand — closed as **hand-free**.",
    ),
)

_THRESH_SAID = False


def hand_thresholds(hand: str = "", say: bool = True) -> dict:
    """The thresholds **actually applied** for this hand and their frames — the one reading point.

    Without `DEXENGINE_HAND`, `active_hand()` dies loudly (no default). Values are read from the owner files every
    time — this function holds no numbers. The marker `[THRESH] …` goes to stderr once per process, so the run log
    records which hand and which source judged (a run without the marker counts as invalid).
    """
    global _THRESH_SAID
    hand = hand or active_hand()
    src = case_source()
    out: dict = {}
    for t in HAND_THRESHOLDS:
        val, ok = _sym(t.owner)
        frames = {label: fn() for label, fn in t.consumers}
        dep = sorted(label for label, fr in frames.items() if fr == FRAME_ROBOT)
        out[t.key] = {
            "rule": t.rule,
            "owner": t.owner,
            "value": val,
            "symbol_ok": ok,
            "frames": frames,
            "hand_dependent": bool(dep),
            "dependent_paths": dep,
            "waiver": t.waiver,
            "why": t.why,
        }
    if say and not _THRESH_SAID:
        parts = []
        for k, r in out.items():
            frs = sorted(set(r["frames"].values()))
            parts.append(f"{k}={r['value']}({'|'.join(frs)}{'⚠' if r['hand_dependent'] else ''})")
        # The arm is not printed: no threshold here takes the arm as input, and calling `active_arm()` would make
        # `import cases.taxonomy` newly require DEXENGINE_ARM — a new failure surface for every importer, for one
        # marker. Only places that need the arm (the mount table) call it.
        print(
            f"[THRESH] hand={hand} source={src} ruleset={RULESET_VERSION} " + " ".join(parts),
            file=sys.stderr,
            flush=True,
        )
        _THRESH_SAID = True
    return out


def hand_dependent_report(hand: str = "") -> list[str]:
    """What to re-derive / re-check when a new hand is registered — the tool says it, so nobody has to remember."""
    hand = hand or active_hand()
    th = hand_thresholds(hand, say=False)
    n_thresh = sum(len(r.thresholds) for r in RULES)
    src = case_source()
    n_dep = sum(1 for r in th.values() if r["hand_dependent"])
    out = [
        f"hand {hand}: of {n_thresh} D-rule thresholds, {len(HAND_THRESHOLDS)} can be tied to hand geometry "
        f"— with the contact / wrist source (D35)={src}, **{n_dep} are actually hand-dependent**"
    ]
    if n_dep == 0:
        out.append(
            "  ✓ no hand-dependent threshold with this source — nothing to re-derive. "
            "Hand independence comes from the **frame**, not the value (see the HAND_THRESHOLDS comment)."
        )
    else:
        out.append(
            f"  ⚠ {n_dep} hand-dependent — read the waivers of the [hand-dependent] entries below. "
            "No check stops a run on an empty waiver."
        )
    # The mount offsets are **read** and printed — they are the origin of the robot virtual-wrist frame. D39 measures
    # even the D14′ oracle on MANO, so **no threshold uses this table with the canonical source (mano)**. They are
    # still printed because, should a robot-frame path appear, this value is the size of the per-hand difference.
    try:
        from dexengine.core import mounts as _ms  # noqa: PLC0415

        rows = sorted(
            (h, m.ee_to_wrist_offset[2], m.provisional) for (a, h), m in _ms.MOUNT_REGISTRY.items() if a == active_arm()
        )
        here = next((z for h, z, _ in rows if h == hand), None)
        out.append(
            f"  mount offset z ({active_arm()} combos, owned by dexengine.core.mounts) — applies **only to "
            f"robot-frame paths**:"
        )
        for h, z, prov in rows:
            d = "" if here is None or h == hand else f"  ({abs(z - here) * 1000:.1f} mm from this hand)"
            out.append(f"      {h:14} {z:+.4f} m{'  ★provisional' if prov else ''}{d}")
    except Exception as e:  # noqa: BLE001
        out.append(f"  ✗ could not read the mount registry: {type(e).__name__}: {e}")
    for dep_kind in (True, False):
        for k, r in th.items():
            if r["hand_dependent"] is not dep_kind:
                continue
            head = "hand-dependent" if dep_kind else "hand-free"
            frs = " · ".join(f"{lab}→{fr}" for lab, fr in r["frames"].items())
            out.append(
                f"  [{head}] {r['rule']:10} {k:10} value={r['value']}"
                + ("" if r["symbol_ok"] else "  ✗ could not read the symbol — was it renamed?")
            )
            out.append(f"      measured at: {frs}")
            out.append(f"      {r['why']}")
            if r["waiver"]:
                out.append(f"      waiver: {r['waiver']}")
    return out
