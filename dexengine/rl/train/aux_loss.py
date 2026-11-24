"""Auxiliary fingertip-force loss: an extra optimizer step after each PPO update."""

import torch.nn as nn

from dexengine.rl.agents.models import ResidualPolicy


def install_aux_force_loss(*, agent, args_cli):
    """Patch the PPO update with the auxiliary fingertip-force loss when ``--aux_force_loss_scale`` > 0."""
    _F_aux, _types_aux = _aux_force_loss_scale()

    _aux_loss_scale = getattr(args_cli, "aux_force_loss_scale", 0.0)
    _aux_warmup_iter = getattr(args_cli, "aux_force_warmup_iter", 0)
    _aux_loss_log = [0.0]  # mutable holder for logging
    _aux_target_magnitude_log = [0.0]  # MSE target magnitude
    _aux_call_counter = [0]  # iter counter for H7 warmup logic

    if _aux_loss_scale > 0.0:
        warmup_msg = f"warmup_iter={_aux_warmup_iter}" if _aux_warmup_iter > 0 else "no warmup"
        print(f"  [AUX FORCE] scale={_aux_loss_scale} {warmup_msg} — activating aux head (MSE on next-frame force)")

        _original_update = type(agent)._update  # unbound

        def _patched_update(self_agent, timestep, timesteps):
            # 1. Run standard PPO update (policy + value loss unchanged)
            _original_update(self_agent, timestep, timesteps)

            # H7 two-stage training: skip aux update during warmup phase
            _aux_call_counter[0] += 1
            if _aux_call_counter[0] <= _aux_warmup_iter:
                if _aux_call_counter[0] == _aux_warmup_iter:
                    print(f"  [AUX FORCE] warmup complete at iter {_aux_call_counter[0]} — activating aux loss now")
                return None

            return _aux_forward_backward(self_agent, _F_aux, _aux_loss_log, _aux_loss_scale, _aux_target_magnitude_log)

        agent._update = _types_aux.MethodType(_patched_update, agent)
        print("  [AUX FORCE] PPO _update monkey-patched — aux loss runs after each standard update")
    else:
        print("  [AUX FORCE] aux_force_loss_scale = 0.0 — aux head present in state_dict, loss disabled")
    return _aux_loss_log, _aux_loss_scale, _aux_target_magnitude_log


def _aux_forward_backward(self_agent, _F_aux, _aux_loss_log, _aux_loss_scale, _aux_target_magnitude_log):
    """2. Aux forward+backward."""
    # 2. Aux forward+backward
    # NOTE: SKRL PPO does not store next_states (off-policy field).
    # Shift states within the rollout buffer: states[t] -> states[t+1].
    policy_model = self_agent.policy

    try:
        raw_states = self_agent.memory.get_tensor_by_name("states")
    except Exception as e:  # noqa: BLE001 — the auxiliary loss is optional; a failed read skips it for this update (warned)
        print(f"  [AUX FORCE WARN] failed to read states: {e}")
        return

    if raw_states.dim() < 2:
        print(f"  [AUX FORCE WARN] unexpected states dim: {raw_states.shape}")
        return

    if raw_states.dim() == 2:
        cur_flat = raw_states[:-1]
        nxt_flat = raw_states[1:]
    else:
        cur = raw_states[:-1]
        nxt = raw_states[1:]
        cur_flat = cur.reshape(-1, cur.shape[-1])
        nxt_flat = nxt.reshape(-1, nxt.shape[-1])

    # Apply state preprocessor (read-only, no running stats update)
    cur_pp = self_agent._state_preprocessor(cur_flat.float(), train=False)
    nxt_pp = self_agent._state_preprocessor(nxt_flat.float(), train=False)

    # Phase A1: action-aware aux head. Read actions from the same rollout slice as states.
    # action a[t] is the action that produced state[t+1], so it pairs with state[t].
    cur_action = None
    if getattr(policy_model, "_aux_action_aware", False):
        try:
            raw_actions = self_agent.memory.get_tensor_by_name("actions")
            if raw_actions.dim() == 3:
                cur_act = raw_actions[:-1]
                cur_action = cur_act.reshape(-1, cur_act.shape[-1]).float()
            else:
                cur_action = raw_actions[:-1].float()
        except Exception as e:  # noqa: BLE001 — the auxiliary loss is optional; a failed read skips it for this update (warned)
            print(f"  [AUX FORCE WARN] aux_action_aware=True but failed to read actions: {e}")
            cur_action = None

    policy_model.train(True)

    # MSE on next-frame force xyz (the surviving aux supervision target)
    pred_forces = policy_model.predict_forces(cur_pp, action=cur_action)  # (N, 5, 3)
    tgt_forces = ResidualPolicy.extract_force_target(nxt_pp)  # (N, 5, 3)
    raw_aux_loss = _F_aux.mse_loss(pred_forces, tgt_forces)
    aux_log_metric = tgt_forces.abs().mean().item()

    aux_loss = _aux_loss_scale * raw_aux_loss

    self_agent.optimizer.zero_grad()
    aux_loss.backward()

    if self_agent._grad_norm_clip > 0:
        nn.utils.clip_grad_norm_(policy_model.parameters(), self_agent._grad_norm_clip)

    self_agent.optimizer.step()

    _aux_loss_log[0] = raw_aux_loss.item()
    _aux_target_magnitude_log[0] = aux_log_metric


def _aux_force_loss_scale():
    """The modules install_aux_force_loss needs, imported lazily: (torch.nn.functional, types). That caller wraps the
    PPO learner's _update() with an extra backward pass that trains the aux_force_head to predict next-frame
    fingertip forces (sliced from next_obs), as a separate optimizer step AFTER the standard PPO update, so PPO
    gradients stay untouched and the encoder receives gradient from both objectives.
    """
    import types as _types_aux

    import torch.nn.functional as _F_aux

    return _F_aux, _types_aux
