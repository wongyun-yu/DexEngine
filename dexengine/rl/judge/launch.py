"""Start-up checks of the judge: context rules of the judge standard, PARK and warm-up defaults, the in-training eval
exit reset. Each decision has one owner here.

The judge value comes from the fresh-process judge (FROZEN_REPLAY: a new Isaac process loads the checkpoint and calls
``frozen_eval_snapshot`` once). In that context PARK between judge batches (``dexengine.rl.judge.park``,
DEXENGINE_EVAL_PARK) is on by default: the mass teleport of a batch cut loses the desk contact of lifted objects, which
then fall freely; PARK holds the object at its reset pose at the cut. One warm-up batch (DEXENGINE_FROZEN_STRICT=1,
``frozen_eval_snapshot`` uses args.eval_warmup_batches) is also that context's default. In-training eval hits the
defect from the first batch (it cuts from a training state) and is not a judge; the recipe turns it off
(DEXENGINE_INTRAIN_EVAL in configs/rl/recipe.yaml). On, it is a reference value and the marker says so.

The judge standard covers bimanual and one-hand runs: both are judged in the fresh context (the launcher's judge
mode, driven by scripts/rl/judge_sidecar.sh) and this module sets the defaults for both.

Context ``launch_ctx``: fresh (fresh-process judge) · render (a render on the same path; no judge batches) · intrain
(in-training eval on) · oop (eval runs in a child process) · none (no eval) · evalpy (``dexengine.rl.eval.run``:
rescoring, the old one-hand reference path, dual capture, kinematic replay, pose render; not a judge).

PARK default ``resolve_park`` (the only writer of DEXENGINE_EVAL_PARK):
  fresh and unset -> set 1 and print the default-on marker; fresh and given 0 -> a "given off, outside the judge
  standard" marker; evalpy and given 1 -> a "not a judge" marker (an explicit choice is accepted); any other context
  and unset -> 0 (nothing written: the PARK module reads unset as 0).
Warm-up default ``resolve_warmup`` (the only owner of DEXENGINE_FROZEN_STRICT's default; read by
``frozen_eval_snapshot``): fresh and unset -> set 1 with a default-on marker; fresh and given 0 -> an "outside the
judge standard" marker; any other context and unset -> 0.

Combination rules ``check_launch`` (once at start-up; a violation dies):
  intrain and PARK=1 and EXIT_RESET=0 -> dies: training would resume from the PARK-ed el=0 state.
  (oop|none) and EXIT_RESET=1         -> dies: this process has no in-training eval exit (nothing consumes it).
  (none|render) and PARK=1            -> dies: no judge batches (a render goes to the record-session seam).
  evalpy and EXIT_RESET=1             -> dies: no training resumes.
  (fresh|render) and EXIT_RESET=1     -> passes; the exit is skipped and a marker printed (no training resumes), so a
                                         post-hoc judge or render that inherits the training settings does not die.
  not fresh and DEXENGINE_JUDGE_CONTACT_TERM or DEXENGINE_JUDGE_OBJTRACK_TERM set -> dies (explicit judge exceptions;
                                         only the fresh-process judge consumes them).
Judge termination rule ``check_judge_terms`` (fresh, after the trainer arguments are parsed): the contact termination
argument must equal ``terms.contact_term()`` (standard or explicit exception), else it dies. The one-hand judge path
inherits the training flags, so the launcher overrides that value; bimanual COLL_PRESET already carries 0. Marker
``[JUDGE_TERM]``.
Markers: ``[EVAL_LAUNCH] ctx=... PARK=... EXIT_RESET=...[ · FROZEN_STRICT=...]`` (start-up; the warm-up field only in
fresh, intrain and oop, which have frozen eval batches) · ``[INTRAIN_EVAL] ...`` (in-training eval on/off; its first
word, off | ⚠ on, is read back from training logs by ``dexengine.rl.judge.record.intrain_eval``) ·
``[EVAL_EXIT_RESET] ...`` (the exit).

DEXENGINE_EVAL_EXIT_RESET (default 0): a real full reset at the in-training eval exit (``exit_reset``). Off, skrl's
``_reset_once`` was already consumed by the last eval batch, ``wrapped.reset()`` does nothing and the end-of-eval state
becomes the training rollout. On, traj_mgr.reset (RSI) resamples and el is set to 0. What remains: the skrl trainer
builds one transition after the eval from the pre-eval observation (the eval runs inside post_interaction and this
layer cannot fix it).
"""

from __future__ import annotations

import os

from dexengine.rl.judge import terms as judge_terms  # judge termination rules (a dependency-free module)
from dexengine.rl.judge.park import ENV_VAR as PARK_VAR  # the PARK module owns the variable name

EXIT_VAR = "DEXENGINE_EVAL_EXIT_RESET"
STRICT_VAR = "DEXENGINE_FROZEN_STRICT"  # warm-up switch: 1 = frozen_eval_snapshot uses args.eval_warmup_batches (1)
CTXS = ("intrain", "oop", "fresh", "render", "none", "evalpy")
_FROZEN_BATCH_CTXS = ("fresh", "intrain", "oop")  # contexts whose process (or child) runs frozen eval batches


def _flag01(name: str, tag: str, env=None) -> bool:
    env = os.environ if env is None else env
    v = (env.get(name, "0") or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"[{tag}] {name}={v!r} — only 0 or 1")
    return v == "1"


def exit_reset_on() -> bool:
    """Read DEXENGINE_EVAL_EXIT_RESET (0/1 only)."""
    return _flag01(EXIT_VAR, "EVAL_EXIT_RESET")


def launch_ctx(replay: bool, eval_on: bool, oop: bool, *, render: bool) -> str:
    """Start-up context of the training entry: fresh-process judge (FROZEN_REPLAY, bimanual and one-hand), its render
    (FROZEN_RENDER), no eval, OOP parent (eval in a child) or in-training eval. The eval entry passes evalpy itself."""
    if replay:
        return "render" if render else "fresh"
    if not eval_on:
        return "none"
    return "oop" if oop else "intrain"


def resolve_park(ctx: str, env=None) -> tuple[bool, str]:
    """The PARK default: (on, source) with source "default" | "given". Fresh and unset writes 1 into env, so every
    ``park.enabled()`` of this process and its children see the same value."""
    env = os.environ if env is None else env
    raw = (env.get(PARK_VAR, "") or "").strip()
    if raw == "":
        if ctx == "fresh":
            env[PARK_VAR] = "1"
            print(
                f"[EVAL_PARK] default on — judge standard of the fresh-process judge ({PARK_VAR}=1). "
                f"To turn it off give {PARK_VAR}=0",
                flush=True,
            )
            return True, "default"
        return False, "default"
    if raw not in ("0", "1"):
        raise ValueError(f"[EVAL_PARK] {PARK_VAR}={raw!r} — only 0 or 1")
    if ctx == "fresh" and raw == "0":
        print(
            f"[EVAL_PARK] ⚠ given off ({PARK_VAR}=0) — outside the judge standard; this value is for comparison, "
            "not a judge",
            flush=True,
        )
    if ctx == "evalpy" and raw == "1":
        print(
            f"[EVAL_PARK] ⚠ given on in the eval entry ({PARK_VAR}=1) — this context is not a judge (rescoring, "
            "reference, replay). Judge = the fresh-process PARK judge (bimanual and one-hand)",
            flush=True,
        )
    return raw == "1", "given"


def resolve_warmup(ctx: str, env=None) -> tuple[bool, str]:
    """The warm-up switch (DEXENGINE_FROZEN_STRICT) default: (on, source). Fresh and unset writes 1 into env, so the
    judge standard's one warm-up batch comes from the context (the one-hand judge gets it too). The only other writer
    is the bimanual launcher environment, which gives 1 explicitly (so the bimanual in-training reference path also
    warms up); in fresh it is the same value and the marker just says "given". Neither can go: without the launcher
    value the bimanual reference path would warm up 0 batches, without this default the one-hand judge would (the
    sidecar's warm-up = 1 row check catches that)."""
    env = os.environ if env is None else env
    raw = (env.get(STRICT_VAR, "") or "").strip()
    if raw == "":
        if ctx == "fresh":
            env[STRICT_VAR] = "1"
            print(
                f"[EVAL_WARMUP] default on — judge standard of the fresh-process judge (one warm-up batch: "
                f"{STRICT_VAR}=1). To turn it off give {STRICT_VAR}=0",
                flush=True,
            )
            return True, "default"
        return False, "default"
    if raw not in ("0", "1"):
        raise ValueError(f"[EVAL_WARMUP] {STRICT_VAR}={raw!r} — only 0 or 1")
    if ctx == "fresh" and raw == "0":
        print(
            f"[EVAL_WARMUP] ⚠ given off ({STRICT_VAR}=0 — no warm-up) — outside the judge standard; this value is "
            "for comparison, not a judge",
            flush=True,
        )
    return raw == "1", "given"


def check_launch(ctx: str, env=None) -> str:
    """Once at start-up: PARK default, combination rules (module docstring), marker. A violation dies. Returns the
    context (check_judge_terms uses the same value)."""
    env = os.environ if env is None else env
    if ctx not in CTXS:
        raise ValueError(f"[EVAL_LAUNCH] start-up context {ctx!r} — one of {CTXS}")
    park, src = resolve_park(ctx, env)
    warm, wsrc = resolve_warmup(ctx, env)
    exit_ = _flag01(EXIT_VAR, "EVAL_EXIT_RESET", env)
    bad = []
    if ctx == "intrain" and park and not exit_:
        bad.append(
            f"{PARK_VAR}=1 in in-training eval needs {EXIT_VAR}=1 — else training resumes from PARK-ed envs (el=0)"
        )
    if ctx in ("oop", "none", "evalpy") and exit_:
        bad.append(f"{EXIT_VAR}=1 but this process has no in-training eval exit (nothing consumes it)")
    if ctx in ("none", "render") and park:
        bad.append(
            f"{PARK_VAR}=1 but this process has no judge batches (nothing consumes it — the fresh-process judge judges)"
        )
    if ctx != "fresh" and judge_terms.OVERRIDE_VAR in env:
        bad.append(
            f"{judge_terms.OVERRIDE_VAR} is set but this context ({ctx}) is not the fresh-process judge "
            "(nothing consumes it)"
        )
    if ctx != "fresh" and judge_terms.OBJTRACK_VAR in env:
        bad.append(
            f"{judge_terms.OBJTRACK_VAR} is set but this context ({ctx}) is not the fresh-process judge "
            "(nothing consumes it)"
        )
    if bad:
        raise RuntimeError(f"[EVAL_LAUNCH] start-up combination refused (ctx={ctx}): " + " · ".join(bad))
    if ctx == "none":
        # The first word after the tag (off, ⚠ on) is read back from training logs (judge/record.py).
        print(
            "[INTRAIN_EVAL] off — no in-training eval. Judge = the fresh-process PARK judge "
            "(scripts/rl/judge_sidecar.sh)",
            flush=True,
        )
    elif ctx in ("intrain", "oop"):
        print(
            "[INTRAIN_EVAL] ⚠ on — a reference value (outside the judge standard: it cuts from a training state and "
            "hits the defect from the first batch). Judge = the fresh-process PARK judge",
            flush=True,
        )
    elif ctx == "render":
        print("[EVAL_LAUNCH] render — not a judge (no judge batches, nothing consumes PARK)", flush=True)
    elif ctx == "evalpy":
        print(
            "[EVAL_LAUNCH] eval entry (rescoring, old one-hand reference path, dual capture, replay, render) — not a "
            "judge · PARK off by default. Judge = the fresh-process PARK judge (bimanual and one-hand, "
            "scripts/rl/judge_sidecar.sh)",
            flush=True,
        )
    e_txt = (
        ("1(fresh process — no training resumes, exit skipped)" if ctx in ("fresh", "render") else "1")
        if exit_
        else "0"
    )
    w_txt = f" · {STRICT_VAR}={int(warm)}({wsrc})" if ctx in _FROZEN_BATCH_CTXS else ""
    print(f"[EVAL_LAUNCH] ctx={ctx} · {PARK_VAR}={int(park)}({src}) · {EXIT_VAR}={e_txt}{w_txt}", flush=True)
    return ctx


def check_judge_terms(ctx: str, contact_term_arg, env=None) -> None:
    """Proof of the fresh-process judge's termination rule: the trainer argument --enable_contact_termination must
    equal the judge standard (explicit exception included, terms.contact_term), else it dies. Marker
    ``[JUDGE_TERM] ...``. Outside fresh it does nothing (check_launch refuses the exception inputs there)."""
    if ctx != "fresh":
        return
    judge_terms.objtrack_term(os.environ if env is None else env)  # value check only (dies before the env is built);
    # application and marker happen at the cfg stage
    want, src = judge_terms.contact_term(os.environ if env is None else env)
    got = int(contact_term_arg or 0)
    if got != want:
        what = "judge standard" if src == judge_terms.STANDARD else "explicit exception"
        raise RuntimeError(
            f"[JUDGE_TERM] contact termination argument {got} of the fresh-process judge != {what} "
            f"{want} — the judge path did not carry the judge termination rule (one-hand: the "
            "launcher's judge mode; bimanual: COLL_PRESET)"
        )
    print(judge_terms.marker(got, src), flush=True)


def exit_reset(wrapped, raw_env, replay: bool) -> None:
    """In-training eval exit (DEXENGINE_EVAL_EXIT_RESET=1): a real full reset, so training resumes through
    traj_mgr.reset (RSI). The reset is proven by ``_reset_once`` being consumed and el == 0 in every env, else it dies.
    replay (fresh-process judge): no training resumes, so only ``wrapped.reset()`` (a no-op) runs and the skip is
    printed."""
    if replay:
        wrapped.reset()
        print("[EVAL_EXIT_RESET] skipped — fresh-process judge (no training resumes · after aggregation)", flush=True)
        return
    tm = raw_env.trajectory_manager
    wrapped._reset_once = True
    wrapped.reset()
    n = int(raw_env.num_envs)
    el0 = int((raw_env.episode_length_buf == 0).sum())
    if wrapped._reset_once or el0 != n:
        raise RuntimeError(
            f"[EVAL_EXIT_RESET] training-resume reset not proven — _reset_once={wrapped._reset_once} "
            f"(should be consumed) · el==0 {el0}/{n}"
        )
    sf = int((tm.start_frames > 0).sum())
    print(
        f"[EVAL_EXIT_RESET] training-resume reset {el0} envs (of {n} · RSI {float(tm.rsi_fraction):.3f} · "
        f"mid-clip starts {sf}/{n} · adaptive RSI record skipped)",
        flush=True,
    )
