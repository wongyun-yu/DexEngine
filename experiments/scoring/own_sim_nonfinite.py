"""One rule for every baseline: a run whose OWN simulator went non-finite scores 0 for that run as a method failure, and
that run is not re-run (re-drawing until a finite run would bias the result; dropping it would inflate the mean).

The trigger is narrow and structural, measured from the method's own outputs (never a word in a log):
  (A) trajectory: the record's reader measured the method's own output trajectory (``own_traj`` — e.g. SPIDER
      0/trajectory_mjwp.npz, ``np.isfinite`` on the object-pose columns ``score.py`` reads) with ≥ 1 non-finite step,
      the method's own process ended rc 0 (``method_rc``) and the run's MATCH_PHYS runtime marker passed (``match_ok``)
      — the method ran normally and saved a non-finite pose, which the scorer then refused;
  (B) empty rollout: the shared scorer's ``empty_rollout`` evidence (attempts ≥ 1 with ≥ 1 non-finite) with 0 scored
      steps (EgoEngine: its attempt records are the method's own output).
Only a FAIL record can be zeroed. A FAIL from a port, the scorer or the infrastructure (rc ≠ 0, a missing marker, no
trajectory, a finite trajectory with a failed score) stays FAIL and is fixed and run again. Used by the Tab. 2 table
build.
"""

OWN_SIM_NONFINITE_NOTE = {
    "empty_rollout": (
        "0 (method failure): the method's own simulator went non-finite in {nf} of {att} attempts and "
        "committed no window — scored 0, not run again (own_sim_nonfinite_zero)"
    ),
    "trajectory": (
        "0 (method failure): the method's own simulator went non-finite — {bad} of {steps} steps of its own "
        "trajectory ({file}) have a non-finite object pose, first at step {first}; the method ended rc 0 — "
        "scored 0, not run again (own_sim_nonfinite_zero)"
    ),
}
# the score a zeroed record carries when the scorer wrote none (class A): 0 on every table measure, one trajectory
ZERO_SCORE = {
    "add_auc": 0.0,
    "add_auc_best": 0.0,
    "add_auc_best_world": 0,
    "spider_sr": 0.0,
    "maniptrans_sr": 0.0,
    "worlds": 1,
    "steps_scored": 0,
    "steps_missing": None,
}


def own_sim_nonfinite_zero(r):
    """None, or (class, note) when record r is a run of the method-non-finite class (see the module doc)."""
    if r.get("verdict") != "FAIL":
        return None
    t = r.get("own_traj") or {}
    if int(t.get("nonfinite_obj_steps") or 0) >= 1 and r.get("method_rc") == 0 and r.get("match_ok") is True:
        return "trajectory", OWN_SIM_NONFINITE_NOTE["trajectory"].format(
            bad=t["nonfinite_obj_steps"], steps=t.get("steps"), file=t.get("file"), first=t.get("first_nonfinite_step")
        )
    s = r.get("score") or {}
    e = s.get("empty_rollout") or {}
    if (
        r.get("score") is not None
        and s.get("steps_scored") == 0
        and int(e.get("attempts") or 0) >= 1
        and int(e.get("nonfinite_attempts") or 0) >= 1
    ):
        return "empty_rollout", OWN_SIM_NONFINITE_NOTE["empty_rollout"].format(
            nf=e["nonfinite_attempts"], att=e["attempts"]
        )
    return None


def apply(r, run):
    """Zero record r in place when it is of the class; print the marker. Returns True when zeroed."""
    z = own_sim_nonfinite_zero(r)
    if z is None:
        return False
    cls, note = z
    if r.get("score") is None:
        r["score"] = dict(ZERO_SCORE)
    r.update(verdict="PASS", zero_decided=note, own_sim_nonfinite=cls)
    print(f"[NONFINITE_ZERO] {r['method']} {r['clip']} {run}", flush=True)
    return True
