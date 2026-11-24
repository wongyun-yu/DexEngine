"""ManipTrans rows that run with the authors' aggregateMode switched off (the one owner of that clip list).

With aggregates on (the authors' ResDexHand.yaml env.aggregateMode=3), object 099 (121 collision shapes: 107 V-HACD
hulls from two needle-thin pieces) and about 1360 or more envs, Isaac Gym's GPU pipeline corrupts the heap during env
creation (create_actor -> free(): invalid pointer). aggregateMode=0 is a value the authors' code supports
(dexhandmanip_bih.py: begin_aggregate only when >= 3, end_aggregate only when > 0; 1 and 2 would call end without
begin). begin_aggregate(..., True) keeps self-collision inside the group, so the aggregate is only a broad-phase
grouping: the collision groups, filters and contact pairs are the same. ROWS = every residual row whose objects include
099 (the only object above 16 shapes among the 64 clips).

rewrite() adds the Hydra override task.env.aggregateMode=0 to the training call and to the evaluation call of a row
command and prefixes the command with an [AGG_OVERRIDE] marker line (train.py prints the resolved cfg and writes
runs/<exp>/config.yaml). The table builder reads ROWS and OBJECT for its notes.
"""

ROWS = (
    "brush_brush_cup__20231026_012",
    "empty_bowl_cup__20230930_124",
    "empty_cup_bowl__20230930_122",
    "pour-in-some_cup_bowl__20230930_129",
)
OBJECT, REASON = "099", "physx_gpu_heap_corruption"
EXPIRY = "until_isaacgym_fix_or_permanent_while_baseline_frozen"
OVR = "task.env.aggregateMode=0"
# training call · evaluation call (maniptrans/gen_cmds.sh writes both once per row)
CALLS = ("main/rl/train.py task=ResDexHand", "experiments/baselines/maniptrans/eval_record.py ")


def marker(c: str) -> str:
    return f'echo "[AGG_OVERRIDE] row={c} aggregateMode=0 object={OBJECT} reason={REASON} expiry={EXPIRY}" && '


def rewrite(cmd: str, c: str) -> str:
    """The row command of clip <c> with the override on both calls (idempotent; dies on a partial or ambiguous form)."""
    if OVR in cmd:  # already applied: must be exactly the applied form
        if not cmd.startswith(marker(c)) or cmd.count(OVR) != 2:
            raise SystemExit(f"[AGG_OVERRIDE] FAIL {c}: partial override in the row command")
        return cmd
    out = cmd
    for call in CALLS:
        if out.count(call) != 1:
            raise SystemExit(f"[AGG_OVERRIDE] FAIL {c}: '{call}' found {out.count(call)}x (expected 1)")
    a = CALLS[0]
    out = out.replace(a, f"{a} {OVR}")
    # evaluation: the Hydra overrides follow ' -- task=ResDexHand' after the recorder
    i = out.index(CALLS[1])
    j = out.index(" -- task=ResDexHand", i)
    k = j + len(" -- task=ResDexHand")
    out = out[:k] + " " + OVR + out[k:]
    return marker(c) + out
