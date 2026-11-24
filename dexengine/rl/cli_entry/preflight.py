"""Trainer preflight (switch DEXENGINE_PREFLIGHT_ONLY): the scene-input checks the env build runs after Kit, run before
Kit on the CPU; the process then ends without starting Kit. The dry run of a training or judging launch starts every
resolved trainer process this way (`rl.launch.process.preflight`), so a launch the scene build would refuse fails
before a GPU job.

  unset or 0  the trainer runs as usual (the entry does not call `run`)
  1           after every pre-Kit check of the entry (`dexengine/rl/cli.py`): the extra-slot ledger check of the
              checkpoint the process loads, then the extra-slot plan of the run's clips and its spawn-list comparison
              with that ledger (slots on), or the scene check of a ledger-less checkpoint (slots off). It returns the
              `[PREFLIGHT] PASS …` line and the entry exits 0; a refusal raises the error the scene build raises.
              A checkpoint path whose file does not exist yet (a dry run ahead of its checkpoint; a real judge
              launch refuses it before the trainer starts, `launch.dispatch`) has no ledger to read: the plan runs
              under the stated or default extra-slot switch, and the PASS line says the ledger was not compared.
  other       fails.

Standard library only at import: the entry imports this module before AppLauncher. `run` imports numpy (through the
plan), which the entry allows only because Kit never starts in this mode (the rule of `dexengine.rl.preapp` guards what
runs before AppLauncher in a process that starts Kit).
"""

from __future__ import annotations

import os

ENV = "DEXENGINE_PREFLIGHT_ONLY"
MARK = "[PREFLIGHT]"
PASS = f"{MARK} PASS"


def requested(environ=None) -> bool:
    """Whether this trainer process is a preflight; a value other than 0|1 fails."""
    v = (os.environ if environ is None else environ).get(ENV, "").strip()
    if v not in ("", "0", "1"):
        raise SystemExit(f"{MARK} {ENV}={v!r} — only 0|1")
    return v == "1"


def run(args_cli) -> str:
    """The checks of the module doc, in the order of a real run (ledger value before the env build, then the slots);
    returns the PASS line."""
    from dexengine.rl.env_setup import extra_slots_ledger as EL
    from dexengine.rl.env_setup import extra_slots_plan as XP

    ctx = f"preflight {args_cli.task}"
    ckpt = EL.ledger_checkpoint(args_cli)
    absent = bool(ckpt) and not os.path.isfile(ckpt)  # no file, no ledger: not read as 'trained with slots off'
    EL.extra_slots_ledger_check(None if absent else ckpt, ctx)  # writes DEXENGINE_EXTRA_SLOTS from the ledger
    if EL.extra_slots_requested():
        plan = XP.plan_rows(getattr(args_cli, "sequence_dir", None), args_cli.sequence)
        XP.record_spawn_list(plan, plan.rule(), ctx)
        what = f"extra slots on · clips {len(plan.seqs)} · slot rows {sum(len(r) for r in plan.rows)}"
    else:
        XP.slots_off(args_cli, ctx)
        what = "extra slots off"
    led = f"checkpoint {ckpt} absent — ledger not compared" if absent else f"checkpoint {ckpt or '-'}"
    return f"{PASS} · {what} · {led} · Kit not started"
