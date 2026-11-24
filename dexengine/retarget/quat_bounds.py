"""How far from unit length a keypoint archive's quaternions may be — one statement, read by the converter
(which repairs an object block outside the bound) and by IK (which refuses an archive outside it).

Hand keys are our own forward-kinematics output, unit to float precision. Object keys carry the source
dataset's quaternions as they are, and those are a little off-unit by nature (TACO's grid-50 originals reach
0.7 %), so the object bound is a corruption bound rather than a unit test.
"""

HAND_QUAT_ATOL = 1e-3
OBJ_QUAT_ATOL = 0.05
