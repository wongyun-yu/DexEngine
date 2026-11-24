"""One owner for binding the entry module's RunContext into a runner module (train.runner, eval.run).

The runners read the entry's fields as module globals (``args_cli``, ``create_env``, ...); each calls
``bind_run_context(globals(), ctx)`` at its start instead of keeping its own copy of the field list.
"""

from __future__ import annotations


def bind_run_context(ns: dict, ctx) -> None:
    """Copy the RunContext fields into the module namespace ``ns`` (the runner's globals())."""
    ns["args_cli"] = ctx.args_cli
    ns["create_env"] = ctx.create_env
    ns["GYM_ID_MAP"] = ctx.gym_id_map
    ns["_ACTIVE_OBJ_KEY"] = ctx.active_obj_key
    ns["_active_side"] = ctx.active_side
    ns["load_agent_yaml"] = ctx.load_agent_yaml
    ns["_residual_act_name"] = ctx.residual_act_name
