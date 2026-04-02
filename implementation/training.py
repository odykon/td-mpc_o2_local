"""
training.py
-----------
Decoder training for latent action space control.

Provides:
    action_decoder_DDPG_update — DDPG-style policy gradient update for the
                                 action decoder, exposed as a method on
                                 DCEM_TDMPC.

Training procedure
------------------
The decoder is trained by maximising the Q-value of the action sequence it
produces, analogous to the actor update in DDPG:

    1. Encode observation -> latent state z (detached: encoder does not train).
    2. Decode u_mean -> action sequence (gradients flow through decoder).
    3. Estimate value using the frozen TD-MPC critic.
    4. Backpropagate -value through the decoder.

The world model (TOLD) must be frozen by the caller before invoking this
function. In the training loop this is done with:

    self.model.track_TOLD_grad(False)
    loss = self.action_decoder_DDPG_update(obs, u_mean, horizon)
    self.model.track_TOLD_grad(True)

u_mean is obtained from DCEMethod with update_mode=True, which keeps the
computation graph alive through the LML scores. The full gradient path is:

    decoder weights -> decode_sequence -> estimate_value -> -loss
"""



def action_decoder_DDPG_update(self, obs, u_mean, horizon):
    """
    Perform one DDPG-style decoder update step.

    'self' here refers to the DCEM_TDMPC instance, giving access to
    self.model, self.estimate_value, and self.action_dec_optim.

    Args:
        obs:     [B, obs_dim] observation batch from the replay buffer.
        u_mean:  [B, latent_action_dim] differentiable latent action mean
                 obtained from DCEMethod(update_mode=True).
        horizon: int planning horizon.

    Returns:
        Scalar decoder loss value (for logging).
    """
    self.action_dec_optim.zero_grad()

    # Detach encoder: gradients flow only through the decoder
    z = self.model.h(obs).detach()

    sequence = self.model.decode_sequence(u_mean, z)
    value    = self.estimate_value(z, sequence, horizon).nan_to_num(0)
    cost     = -value.mean()

    cost.backward()
    self.action_dec_optim.step()

    return cost.item()
