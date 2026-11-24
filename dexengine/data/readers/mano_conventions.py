"""MANO left-hand model per dataset — the one statement. No imports on purpose: the raw readers
(``dexengine.data.readers.hot3d`` · ``taco`` / ``mano_lib`` · ``hocap``) and the scene oracles
(``dexengine/data/readers/raw/*``) both read it; neither restates it. The oracles' independence is their
measurement: ``readers/raw/common.left_convention_check`` holds the statement against the release's own labels or
the delivered keypoints and dies when the other reading fits better.

The published ``MANO_LEFT.pkl`` carries the smplx #48 defect: its first shape direction ``shapedirs[:, 0, :]`` equals
the right model's (sum of absolute differences < ``LEFT_SD0_DEFECT_SUM``). smplx loads the pkl as it is (it does not
negate anything). Whether a dataset's hand fits were made with that column negated is a fact of the dataset, measured
against an independent reference on the same frames (left error median, the right hand as the control):
  hot3d    negated — the UmeTrack landmarks of the same timestamps: 5.9–8.2 mm negated vs 8.6–47.7 mm as published,
           right 6.2–8.6 mm (13 subjects · 1,560 frames per hand · 1560/1560 frames favour negated). The HOT3D toolkit
           ``mano_layer.py`` negates it too.
  arctic   as published — the subject's SMPL-X body fit hand joints: 2.7–4.1 mm vs 3.7–31.5 mm negated, right
           2.6–3.2 mm; the delivered keypoints 0.8–2.5 mm vs 3.5–39 mm. ARCTIC's ``construct_layers`` does not negate.
  oakink2  as published — the SMPL-X body fit hand joints: 2.2–3.0 mm vs 3.1–6.3 mm negated (1440/1440 frames);
           the delivered keypoints 0.5–2.1 mm vs 1.3–5.5 mm. OakInk2's manotorch layer does not negate.
  hocap    as published — the release's own ``hand_joints_3d`` labels: 0.000 mm as published (hocap._joints21).
  taco     as published — no independent labels in the release; the official loader (manopth) does not negate.
  grab     as published — GRAB gives no betas (the shape is the subject's template), so the column is multiplied by 0;
           stated for completeness, it cannot change a GRAB hand.
A dataset not listed has no statement: asking for it raises.
"""

#: sum |left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :]| below this = the published (defective) left model
#: (the HOT3D toolkit's test, ``< 1``)
LEFT_SD0_DEFECT_SUM = 1.0
#: dataset -> its left-hand fits were made with the left model's shapedirs[:, 0, :] negated
LEFT_FIX = {"hot3d": True, "arctic": False, "oakink2": False, "hocap": False, "taco": False, "grab": False}


def left_fix(dataset: str) -> bool:
    """The statement for `dataset` (raises for a dataset without one)."""
    if dataset not in LEFT_FIX:
        raise KeyError(
            f"mano_conventions: no left-hand statement for dataset {dataset!r} (LEFT_FIX: {sorted(LEFT_FIX)})"
        )
    return LEFT_FIX[dataset]


def left_negate(dataset: str, sd0_sum: float) -> bool:
    """Whether to negate the loaded left model's shapedirs[:, 0, :] for `dataset`, given the measured
    sum |left.sd[:, 0] - right.sd[:, 0]| of the two loaded models. A negating dataset leaves an already corrected
    model alone (the toolkit's rule); a dataset fitted with the published model cannot be served by a corrected one."""
    fix = left_fix(dataset)
    defective = sd0_sum < LEFT_SD0_DEFECT_SUM
    if not fix and not defective:
        raise ValueError(
            f"mano_conventions: {dataset} was fitted with the published MANO_LEFT.pkl, but the loaded "
            f"left model is already corrected (sum {sd0_sum:.4f} >= {LEFT_SD0_DEFECT_SUM}) — load the "
            "published pkl"
        )
    return fix and defective


def negate_left(dataset: str, left_sd, right_sd) -> bool:
    """Apply the statement in place to a loaded left model's shapedirs (numpy or torch, (778, 3, B)); True = negated."""
    flip = left_negate(dataset, float(abs(left_sd[:, 0, :] - right_sd[:, 0, :]).sum()))
    if flip:
        left_sd[:, 0, :] *= -1
    return flip
