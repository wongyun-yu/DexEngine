"""Hand tracker (kinematic retargeting solve), `python -m dexengine.retarget.ik`.

Modules: `inputs` (archive checks, entity reference table), `model` (IK model assembly, limit sets), `posture` (the
posture prior as joint forces), `solve` (per-frame settle loop), `post` (smoothing, object passthrough, debug
rollout), `branch` (finger branch windows after the solve), `outputs` (archives and diagnostics), `__main__` (CLI).
`prior` (posture-prior rules) and `hyperext_limits` (the same joint cut outside the solver) are the shared prior
modules.
"""
