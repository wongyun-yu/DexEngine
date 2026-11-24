"""Shared helpers for MANO + robot contact annotation (thresholds, smoothing)."""

from __future__ import annotations

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
REGIONS = ["fingertip", "whole_finger"]
# threshold sweep saved in contact_bin: 0.25,0.5,0.75,1.0,1.25,1.5 cm
THRESHOLDS_M = np.array([0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015], dtype=np.float32)

# Chosen operating point (see threshold diagnostics): 1.0 cm + close gaps <= 5 frames
CONTACT_THRESHOLD_M = 0.01
# keypoint (joint centre) contact adds about the skin thickness; a 1 cm threshold wrongly gives 0 contact for real
# grasping hands (dexcanvas 34/47, many oakink2; measured)
KEYPOINT_PAD_M = 0.01
SMOOTH_MAX_GAP = 5  # fill off-runs <= this between two on-runs (morphological close)
SMOOTH_MIN_ON = 1  # drop on-runs <= this (remove isolated 1-frame blips, morphological open)


def runs(binary):
    """Maximal contiguous True runs -> list of (start, end_exclusive)."""
    b = np.asarray(binary, int)
    e = np.diff(np.r_[0, b, 0])
    return list(zip(np.where(e == 1)[0].tolist(), np.where(e == -1)[0].tolist(), strict=True))


def smooth_contact(b, max_gap=SMOOTH_MAX_GAP, min_on=SMOOTH_MIN_ON):
    """Close short off-gaps (<=max_gap) between on-runs, then drop short on-runs (<=min_on)."""
    b = np.asarray(b, bool).copy()
    r = runs(b)
    for i in range(len(r) - 1):  # close
        if r[i + 1][0] - r[i][1] <= max_gap:
            b[r[i][1] : r[i + 1][0]] = True
    for s, e in runs(b):  # open
        if e - s <= min_on:
            b[s:e] = False
    return b


def smooth_contact_block(contact_TFR):
    """Apply smoothing per (finger, region) channel of a (T,F,R) bool array."""
    out = np.zeros_like(contact_TFR, bool)
    F, R = contact_TFR.shape[1], contact_TFR.shape[2]
    for fi in range(F):
        for ri in range(R):
            out[:, fi, ri] = smooth_contact(contact_TFR[:, fi, ri])
    return out
