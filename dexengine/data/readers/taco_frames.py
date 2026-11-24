"""TACO capture world ↔ scene (desk) frame — the **single source of truth** for the z shift.

Official convention (settled from the TACO-Instructions code):
  hand = MANO(θ)/1000 + hand_trans, object = R·p + t — both are composed directly into the
  **same world frame** (floor origin, z-up); no desk-height file and no extra alignment exist.

  ※ The MANO reproduction itself is already verified: hand vertices from our smplx path
    (dexengine.record.viz.mano_model.mano_fk_batch) and from TACO's official manopth path
    (use_pca=False, ncomps=45, center_idx=0) agree to within **0.0001 mm**.
    So MANO is innocent; the only thing that can be wrong is the **frame constant** below.

Our chain's copy of the object poses is in the 'desk origin' frame (rebased), so we need a
constant that lowers the hand into the same frame:

  hand_scene = hand_capture − [0, 0, hand_shift_z(seq)]

The value can differ per session (capture date) — TACO ships a separate world coordinate
system visualization per date (assets/world_coordinate_system_visualizations). Measured
(lower tail of the raw MANO vertex z = the desk surface):

    23 of the 24 sessions : desk z = 0.5287 ~ 0.5503  (−18.9 ~ +2.7 mm vs the constant)
    20231002 (5 clips)    : desk z = **−0.1775**      (−725.1 mm vs the constant)

  20231002 is a **session whose world origin is different**. It is not a defect in the source —
  that session's hands and objects are consistent with each other (in that tree the robot
  fingertips sit 0.05~0.13cm from their own object). A re-derivation that handles every session
  with **one global constant** gets it wrong. Cross-checked by 2 independent methods:
      desk-height estimate (raw MANO only)         : −0.1775
      robot-anchor median (raw wrist − robot hand) : −0.1921  (5 clips, −0.175~−0.214)
  → the two estimates differ by 15mm. The adopted value sits inside the contact fit
    and inside the agreement band of the two estimates, and is pinned per session in
    SESSION_SHIFT_Z below.

FORBIDDEN: back-solving z from a per-clip object rest-height constant (e.g. _OBJ_REST_Z=0.5571
in the old render_mano) — rest height differs per object, so that gives +2~4cm of per-clip
error.
"""

import re

import numpy as np

# Capture world → scene (desk) frame: hands_scene = hands_raw + HAND_SHIFT (z only).
# Default = common to 23 sessions. (Original derivation: MANO-wrist fit over 9 overlapping
# sequences, rotation ≤0.101°, spread 2.3mm. Verified: desk-contact anchor per session
# −18.9~+2.7mm, all sessions.)
HAND_SHIFT = np.array([0.0, 0.0, -0.5476])
HAND_SHIFT_Z = float(-HAND_SHIFT[2])  # = 0.5476 (amount to lower, positive)

# Per-session (YYYYMMDD) exception — that session's capture world origin is different. The
# value = that session's capture-world desk z (= amount to lower). Rationale: the docstring
# above.
SESSION_SHIFT_Z: dict[str, float] = {
    # 20231002 — **observed from the official Object_Poses** (mano_algo_audit).
    # That session's capture-world desk z, measured from the official object poses = −0.17653.
    # The indirect estimates (desk height −0.1775 / robot anchor −0.1921 / contact fit −0.187)
    # sit up to 8.47 mm away from the official value.
    # The other 23 sessions match the code constant 0.5476 to ≤3.05 mm → no exception needed.
    "20231002": -0.17653,
}

_SID = re.compile(r"(?<!\d)(\d{8})_\d+")


def session_of(seq_or_sid: str | None) -> str | None:
    """Extract the capture session (YYYYMMDD) from a clip name or a sequence id.

    Accepts all of 'hit_hammer_helmet__20231002_063', '20231002_063',
    'cut_spoon_plate__20231027_089_pl_rt'.
    """
    if not seq_or_sid:
        return None
    m = _SID.search(str(seq_or_sid))
    return m.group(1) if m else None


def hand_shift_z(seq_or_sid: str | None = None) -> float:
    """That clip's capture-world → scene z shift (amount to lower, positive). Honours the session
    exception.

    Without seq_or_sid it returns the corpus default (HAND_SHIFT_Z) — the value callers without a
    clip name get. **New code must pass the clip name.**
    """
    s = session_of(seq_or_sid)
    return SESSION_SHIFT_Z.get(s, HAND_SHIFT_Z) if s else HAND_SHIFT_Z


def hand_shift(seq_or_sid: str | None = None) -> np.ndarray:
    """The (3,) vector used as hands_scene = hands_raw + hand_shift(seq)."""
    return np.array([0.0, 0.0, -hand_shift_z(seq_or_sid)])


def is_session_exception(seq_or_sid: str | None) -> bool:
    return session_of(seq_or_sid) in SESSION_SHIFT_Z
