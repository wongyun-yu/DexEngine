"""Scene, physics and control constants of the RL task (stdlib only; placement code reads it without Isaac).

Override order, lowest to highest: these defaults -> the cfg factories -> launcher CLI flags.
Finger PD gains are per hand (`HandSpec.finger_kp/kd`), arm and head drives and the head pitch per arm
(`ArmSpec.arm_drive` · `head_drive`), not here.
"""

from dexengine.core.desk import DESK_TOP_SURFACE_Z, DESK_X_HANDTRACK
from dexengine.rl.tasks import env_flags

# ---------------------------------------------------------------- scene geometry
DESK_HEIGHT = DESK_TOP_SURFACE_Z  # desk top z (m), owner core.desk; 0.0 would be ground level
DESK_OFFSET_X = DESK_X_HANDTRACK  # desk center x of the hand-tracking scene; task scenes use core.desk.desk_top()

DOME_LIGHT_COLOR = (0.75, 0.75, 0.75)
DOME_LIGHT_INTENSITY = 2500.0

ROBOT_DEFAULT_POS = (-0.65, -0.15, 0.30)  # robot root spawn (m) when the combo does not give one

# ---------------------------------------------------------------- rigid-body physics (train == eval)
RIGID_LINEAR_DAMPING = 20.0  # ManipTrans value
RIGID_ANGULAR_DAMPING = 20.0  # ManipTrans value
MAX_DEPENETRATION_VELOCITY = 5.0  # ManipTrans value; robot and objects use the same cap
GROUND_FRICTION = 0.1  # ManipTrans value

# ---------------------------------------------------------------- finger joints (common physics)
FINGER_EFFORT = 1.0  # N*m
FINGER_VELOCITY = 2.0  # rad/s
FINGER_ARMATURE = 0.01  # sim rotor inertia

# Mimic joints are a solver constraint (PhysxMimicJointAPI), not PD.
MIMIC_KP = 0.0
MIMIC_KD = 0.0
MIMIC_ARMATURE = 0.01

# ---------------------------------------------------------------- residual policy
RESIDUAL_SCALE = 0.3  # fallback: combined = base + residual * RESIDUAL_SCALE
WRIST_RESIDUAL_SCALE = 0.3  # wrist dims (0:6)
FINGER_RESIDUAL_SCALE = 0.3  # finger dims (6:)
ACTION_EMA = 0.4  # EMA factor on the raw residual (ManipTrans value)

# ---------------------------------------------------------------- head camera
# Mounted under the head pitch link (moves with it). Rx(-90 deg) aligns the ROS z-forward optical axis with the
# camera link's +y axis; the position offset centers the sensor on the camera mesh face.
# DEXENGINE_CAMERA_RES sets the square resolution (renders hold every frame in RAM, so video jobs lower it).
CAMERA_WIDTH = int(env_flags.get("DEXENGINE_CAMERA_RES"))
CAMERA_HEIGHT = CAMERA_WIDTH
# DEXENGINE_CAMERA_FOCAL_MM narrows the field of view for videos only; 8.0 mm = the ZED wide angle (~105.7 deg).
CAMERA_FOCAL_LENGTH = float(env_flags.get("DEXENGINE_CAMERA_FOCAL_MM"))
CAMERA_HORIZONTAL_APERTURE = 20.955
CAMERA_FOCUS_DISTANCE = 400.0
CAMERA_CLIPPING_RANGE = (0.05, 10.0)  # near plane 5 cm keeps the camera housing out of the image
CAMERA_OFFSET_POS = (0.0, 0.005, 0.015)  # front face center of the camera mesh
CAMERA_OFFSET_ROT = (0.7071068, -0.7071068, 0.0, 0.0)  # Rx(-90 deg), wxyz
CAMERA_UPDATE_PERIOD = 0.1
