"""Finger-penalty span and scales: the per-side plan, the marker (``FINGER_PENALTY_PATH``), the check it was printed.

The finger penalty (residual L2 and action-rate terms on the finger dimensions) is scaled per hand by
``dexengine.core.hands.finger_penalty_plan``. ``train_cfg_overrides`` applies the plan and prints one marker line
per
side; after ``create_env`` the trainer requires a line for every expected side, so a skipped step stops the run.
"""

import os

from dexengine.core.errors import InputMissingError
from dexengine.core.hands.recipes import SPAN_GIVEN

#: head of the finger-penalty path marker, one line per side (train) or one line (eval path); the judge sidecar
#: (scripts/rl/judge_sidecar.sh) counts the frozen_replay lines of a judge log against its sides
FINGER_PENALTY_PATH = "[FINGER_PENALTY] path="

#: Sides whose finger-penalty marker was printed (filled by _report_finger_penalty, checked by
#: _require_finger_penalty_marks).
_FINGER_MARKS: list = []


def _finger_penalty_span(args_cli) -> tuple[str, str]:
    """(span, source) from ``dexengine.core.hands.resolve_finger_penalty_span`` (the ledger uses the same function)."""
    from dexengine.core.hands import resolve_finger_penalty_span

    return resolve_finger_penalty_span(getattr(args_cli, "finger_penalty_span", None))


def _finger_penalty_path(args_cli) -> str:
    """Launch path named in the marker: frozen_replay (judge), resume (continued training) or train."""
    if os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") == "1":
        return "frozen_replay"
    return "resume" if (getattr(args_cli, "resume", None) or getattr(args_cli, "resume_from", None)) else "train"


def _finger_penalty_sides(args_cli, *, _residual_act_name) -> list[str]:
    """Residual action terms the finger penalty applies to (one per side): both hands, or the single residual term."""
    if bool(getattr(args_cli, "bimanual", 0)):
        return ["right_hand_joint_pos", "left_hand_joint_pos"]
    return [_residual_act_name()]


def _finger_penalty_side(cfg, args_cli, span: str, act_name: str, plan: dict):
    """Finger-penalty plan ``(dims, w_scale, dz_scale)`` of one side's action term, from
    ``dexengine.core.hands.finger_penalty_plan`` with n = this term's finger count (``len(finger_joint_names)``).
    Cached in ``plan[act_name]`` so the marker is printed per side."""
    from dexengine.core.hands import finger_penalty_plan

    if act_name not in plan:
        term = getattr(getattr(cfg, "actions", None), act_name, None)
        if term is None or not hasattr(term, "finger_joint_names"):
            raise InputMissingError(
                f"[FINGER_PENALTY] no action term {act_name} — cannot count the fingers "
                "(no run without the span and scale set)"
            )
        n = len(term.finger_joint_names)
        try:
            fp = finger_penalty_plan(args_cli.hand, span, n)
        except ValueError as e:
            raise SystemExit(f"[FINGER_PENALTY] {act_name}: {e}") from e
        plan[act_name] = {"plan": fp, "n": n, "users": []}
    return plan[act_name]["plan"]


def _finger_penalty_user(plan: dict, act_name: str, name: str, base_w: float, term, base_dz=None) -> None:
    """Record one consuming reward term. The effective weight and deadzone are read back from the term after scaling,
    so the marker shows the applied values. ``base_dz`` = the residual-L2 term's base deadzone (None for the action-rate
    term)."""
    eff_dz = None if base_dz is None else float(term.params["deadzone"])
    plan[act_name]["users"].append((name, float(base_w), float(term.weight), base_dz, eff_dz))


def _report_finger_penalty(cfg, args_cli, span: str, source: str, plan: dict, *, _residual_act_name) -> None:
    """Print the finger-penalty marker, one line per expected side: path, span (source), dims, n, weight scale, deadzone
    scale, and base→effective weight per term (and base→effective deadzone for the residual-L2 term). An explicit span
    with no consuming term raises."""
    path = _finger_penalty_path(args_cli)
    for act_name in _finger_penalty_sides(args_cli, _residual_act_name=_residual_act_name):
        dims, w_scale, dz_scale = _finger_penalty_side(cfg, args_cli, span, act_name, plan)
        p = plan[act_name]
        users = (
            " · ".join(
                f"{t} {b:g}→{e:.6g}" + ("" if bdz is None else f" dz {bdz:g}→{edz:.6g}")
                for t, b, e, bdz, edz in p["users"]
            )
            or "no consuming term (both penalties off)"
        )
        side = act_name.split("_", 1)[0]
        print(
            f"{FINGER_PENALTY_PATH}{path} span={span}({source}) side={side} dims=({dims[0]},{dims[1]}) n={p['n']} "
            f"w_scale={w_scale:.6g} dz_scale={dz_scale:.6g} · {users} (owner hands.recipes.finger_penalty_plan)",
            flush=True,
        )
        _FINGER_MARKS.append(side)
    if source == SPAN_GIVEN and not any(p["users"] for p in plan.values()):
        raise InputMissingError(
            f"[FINGER_PENALTY] --finger_penalty_span {span} given but no term consumes it (both penalties "
            "off) — no run with it unapplied"
        )


def _require_finger_penalty_marks(args_cli, *, _residual_act_name) -> None:
    """After create_env: every expected side printed its penalty marker; a missing one means the finger-penalty step
    was skipped, so this raises."""
    want = [a.split("_", 1)[0] for a in _finger_penalty_sides(args_cli, _residual_act_name=_residual_act_name)]
    missing = [s for s in want if s not in _FINGER_MARKS]
    if missing:
        raise InputMissingError(
            f"[FINGER_PENALTY] marks missing: sides {missing} (marked sides {_FINGER_MARKS}) — the finger "
            "penalty span and scale step was skipped; not going on"
        )
