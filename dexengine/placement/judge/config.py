"""Search conditions of the placement judge (paper Sec. 3.3), read from `configs/placement/judge.yaml`.

Every field is the default of the judge flag with the same name (`vol_frac` <-> `--vol-frac`). The paper's constants
are named here: `vol_frac` (80 % coverage of the screening), `dp_err_mm` (3 cm tracking threshold, measured as the
stretch-weighted p99 of `retime.stretch.weighted_p99`), `random_start_mult` / `random_start_min`
(NS = max(4 * track_seeds, 32) random starts per placement) and `swivel_max_deg`.

No third-party import except PyYAML, so the parser and the input record can read it in any interpreter.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from dexengine.core.repo import CONFIGS_DIR

#: The default configuration file.
CONFIG_FILE = CONFIGS_DIR / "placement" / "judge.yaml"


@dataclasses.dataclass(frozen=True)
class JudgeConfig:
    """One judge configuration. Field order and names follow the YAML file (sections are only grouping)."""

    # grid
    stride: int
    half_cells: int
    half_cells_x: int
    half_cells_y: int
    n_yaw: int
    yaw_lo: float
    yaw_hi: float
    dz_cells: str
    hand_margin: float
    anchors: str
    max_checks: int
    # screening
    vol_frac: float
    prune_radius_mm: float
    # path
    dp_stride: int
    dp_budget_mult: float
    dp_max_stretch: float
    dp_safety: float
    dp_atoms: int
    dp_travel: float
    dp_clr_w: float
    dp_still_boost: float
    dp_err_mm: float
    dp_cov: float
    jump_max: float
    swivel_max_deg: float
    # confirm
    track_seeds: int
    track_reseeds: int
    track_iters: int
    random_start_mult: int
    random_start_min: int
    n_candidates: int
    seed: int
    # exhaustive
    exhaustive: bool
    exhaust_max: int
    exhaust_cands: int
    batch_cells: int
    gate_batch: int
    order_nat: int
    clip_budget_s: float
    # desk
    desk_shrink_robot: float
    desk_obj_frame0: int

    def random_starts(self, track_seeds: int) -> int:
        """Random IK starts per placement for `track_seeds` tracked seeds: max(mult * M, min)."""
        return max(self.random_start_mult * track_seeds, self.random_start_min)


_CAST = {"int": int, "float": float, "str": str, "bool": bool}


def load(path: str | Path | None = None) -> JudgeConfig:
    """Read a judge YAML file (default `configs/placement/judge.yaml`). Unknown or missing keys raise ValueError."""
    p = Path(path) if path else CONFIG_FILE
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    flat: dict = {}
    for section, body in doc.items():
        if not isinstance(body, dict):
            raise ValueError(f"{p}: section {section!r} is not a mapping")
        for k, v in body.items():
            if k in flat:
                raise ValueError(f"{p}: key {k!r} appears twice")
            flat[k] = v
    fields = {f.name: f for f in dataclasses.fields(JudgeConfig)}
    unknown, missing = sorted(set(flat) - set(fields)), sorted(set(fields) - set(flat))
    if unknown or missing:
        raise ValueError(f"{p}: unknown keys {unknown}, missing keys {missing}")
    out = {}
    for k, f in fields.items():
        v = flat[k]
        want = _CAST[f.type]
        if want is bool and not isinstance(v, bool):
            raise ValueError(f"{p}: {k} must be true/false, got {v!r}")
        out[k] = want(v)
    return JudgeConfig(**out)


#: Judge flags whose effective values identify a judge run in the per-clip input record (`input_record.rules` hash).
FINGERPRINT_FLAGS = (
    "--desk-shrink-robot",
    "--exhaustive",
    "--exhaust-max",
    "--batch-cells",
    "--n-candidates",
    "--exhaust-cands",
    "--jump-max",
    "--order-nat",
    "--desk-obj-frame0",
    "--gate-batch",
    "--clip-budget-s",
    "--arm-flip",
)


def _flag_text(v) -> str:
    if isinstance(v, bool):
        return "" if v else "0"
    if isinstance(v, float):
        return format(v, "g")
    return str(v)


def canon_flags(cfg: JudgeConfig | None = None) -> dict[str, str]:
    """{flag: value text} of FINGERPRINT_FLAGS under the configuration (an on/off flag that is on = '')."""
    from dexengine.placement import arm_flip  # noqa: PLC0415 — owner of the --arm-flip default

    c = cfg or load()
    out = {}
    for flag in FINGERPRINT_FLAGS:
        name = flag[2:].replace("-", "_")
        out[flag] = _flag_text(arm_flip.JUDGE_GATE_DEFAULT if name == "arm_flip" else getattr(c, name))
    return out
