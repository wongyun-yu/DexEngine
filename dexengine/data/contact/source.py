"""The contact source read by cases, roles and the active side, and the measured MANO calibration files.

``CONTACT_SOURCE = "mano"``: human-hand (MANO) keypoints against the object surface. It does not depend on the robot
hand, so every hand gets the same cases and roles. Cases record it in ``task_info.case.contact_source`` and the
hand-object assignment in ``task_info.rules.ASSIGN.contact_source``; a reader whose record differs refuses the tree.

The MANO thresholds are measured, not written in code: the fingertip distance (``best_thr_m``) and the palm proxy
coefficient (``alpha``) are JSON files under ``configs/contact/``. This module is dependency-free so the registry and
the light readers can import it.
"""

from __future__ import annotations

from pathlib import Path

CONTACT_SOURCE = "mano"

#: Folder of the measured MANO calibration files (tracked in the repository).
CALIB_DIR = Path(__file__).resolve().parents[3] / "configs" / "contact"
#: Fingertip contact threshold of the MANO source (key ``best_thr_m``, metres).
TIP_DISTANCE_FILE = CALIB_DIR / "contact_source_distance.json"
#: Palm proxy coefficient (key ``alpha``: how far from the wrist towards the fingertips the palm point sits).
PALM_ALPHA_FILE = CALIB_DIR / "mano_palm_alpha.json"
