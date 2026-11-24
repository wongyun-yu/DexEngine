"""How far from unit length a keypoint archive's quaternions may be — one statement, read by the converter, by IK
and by every reader that builds a rotation from a stored object quaternion.

Hand keys are our own forward-kinematics output, unit to float precision. Object keys come from the source dataset,
and those are off-unit by nature (TACO's grid-50 originals reach 5 % within a clip). The converter writes them unit
(mano_lib.repair_object_quaternions, every clip): a reader that builds R from q as read — R(q) = I + |q|²(R(q/|q|) − I)
for the wxyz formula — gets a matrix that is not a rotation, and the object geometry it places is scaled by up to
|q|² − 1 (on the TACO base trees: 1.9 mm in the f0 object window, 22.6 mm later in a clip).
OBJ_QUAT_ATOL is the corruption bound: an object block beyond it is a tracking glitch and the converter marks the
archive `quat_normalized`. UNIT_ATOL is the unit bound every reader requires (require_unit).
"""

import numpy as np

HAND_QUAT_ATOL = 1e-3
OBJ_QUAT_ATOL = 0.05
UNIT_ATOL = 1e-6
#: How a tree written before that normalisation is rebuilt (docs/datasets.md, TACO): the old base clip folder is moved
#: aside so no old file mixes in, then stage 1 (lift → converter → IK) and the base layer run again.
REBUILD = (
    "move data/<ds>/base/<hand dir>/<emb>/<clip> aside, then run "
    "`python scripts/stage1_retarget.py --bases <clips.txt> --arm <arm> --hand <hand> --dataset <ds> --emb <emb>` and "
    "`python dexengine/data/base_adapter.py --dataset <ds> --hand <hand> --clips <clip,...>` (docs/datasets.md, TACO)"
)


def require_unit(quats, where: str) -> None:
    """Fail when any quaternion of `quats` (..., 4) is off unit length by more than UNIT_ATOL (an off-unit
    object quaternion places a scaled object; a tree written before the converter normalised them dies here)."""
    q = np.asarray(quats, dtype=np.float64)
    if not q.size:
        return
    dev = float(np.abs(np.linalg.norm(q, axis=-1) - 1.0).max())
    if not dev <= UNIT_ATOL:
        raise ValueError(
            f"[QUAT_UNIT] {where}: object quaternion off unit length by {dev:.3g} > {UNIT_ATOL} — the tree predates "
            "the converter's normalisation (dexengine.data.readers.mano_lib.repair_object_quaternions): rebuild it — "
            f"{REBUILD}"
        )
