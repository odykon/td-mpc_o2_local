"""
implementation/pg_training.py

On-policy policy-gradient training functions for the action decoder.
These are designed to be bound as methods on DCEM_TDMPC (or patched onto
TDMPC with setattr), so every function takes `self` as first argument.

Bugs fixed vs. the original pg.py
----------------------------------
1. action_entropy_loss: `agent.cfg.action_shape[0]`  →  `self.cfg.action_dim`
2. action_entropy_loss: decoded_seq reshape now uses permute first so the
   dimensions map correctly:
       [horizon, B*S, action_dim]  →  [B, horizon, S, action_dim]
"""

import torch
from torch.distributions import MultivariateNormal
import torch.nn.utils as utils


# ---------------------------------------------------------------------------
# 1. Policy-Gradient decoder update with a learned value baseline
# ---------------------------------------------------------------------------

def PG_withV(self, obs, u_mean, u_std, reward, obs_t1, original_action, next_rewards, next_obses, alpha_v, horizon):

    #MC estimate and Advantage Calculation
    gamma = 0.99
    discounted_rewards = 0

    with torch.no_grad():
        for t in range(1):
            discounted_rewards += next_rewards[:,t] * (gamma ** t)
        final_obs = next_obses[:,t]
        z_final = self.model.h(final_obs) #[batch_size , latent_dim]
        V_final = self.model._V(z_final).squeeze(-1) #[batch_size]
        Q_estimate = discounted_rewards + (gamma**(t+1)) * V_final
        
    z_0 = self.model.h(obs)
    V_z = self.model._V(z_0).squeeze(-1)

    advantage = Q_estimate - V_z.detach()
    sequence = self.model.decode_sequence(original_action.detach(), z_0)
    action = sequence[0, :].squeeze(0)
    #Reward_loss = self.estimate_value(z_0, sequence, horizon).nan_to_num(0)

    #Q_loss = torch.min(*self.model.Q(z_0, action))
    x = torch.cat([z_0, action], dim=-1)
    Reward_loss = self.model._reward(x)


    latent_dist = torch.distributions.Normal(loc=u_mean, scale=u_std)
    log_probs_per_dim = latent_dist.log_prob(original_action)  # [batch_size, 128]
    current_log_probs = log_probs_per_dim.mean(dim=1)  # [batch_size]

    #latent_entropy = latent_dist.entropy().mean(dim=1)

    #Variance_loss = self.action_variance_loss(u_mean, u_std, z_0)
    Entropy_loss = self.action_entropy_loss(u_mean, u_std, z_0, num_samples = 20)
    V_loss = self.V_net_update(reward, obs, obs_t1)

    """Decoder update"""
    PG_loss = current_log_probs * advantage #[batch_size]

    variance_coeff = alpha_v
    Decoder_loss = -(PG_loss + self.cfg.dec_reward_coeff * Reward_loss + variance_coeff* Entropy_loss).mean()
    self.action_dec_optim.zero_grad()
    Decoder_loss.backward()
    utils.clip_grad_norm_(self.model._action_decoder.parameters(), max_norm=1)
    self.action_dec_optim.step()

    return {
        'Decoder_loss': Decoder_loss,
        'V_loss': V_loss,
        'PG_loss': PG_loss.mean(),
        'Reward_loss': Reward_loss.mean(),
        'Entropy_loss': Entropy_loss.mean(),
        'Advantage': advantage.mean(),
        'current_log_probs': current_log_probs.mean(),
    }


# ---------------------------------------------------------------------------
# 2. Value network update (TD(0) with bootstrapped target)
# ---------------------------------------------------------------------------

def V_net_update(self, reward, obs, next_obs):
    """
    One-step TD update for the value network _V.

    Args:
        reward   (Tensor): [B]
        obs      (Tensor): [B, obs_dim]
        next_obs (Tensor): [B, obs_dim]

    Returns:
        Tensor: scalar MSE loss (already backward'd)
    """
    gamma = 0.99

    with torch.no_grad():
        z_next  = self.model.h(next_obs)
        V_next  = self.model._V(z_next).squeeze(-1)   # [B]
        target  = reward + gamma * V_next              # [B]

    z_0    = self.model.h(obs)
    V_z    = self.model._V(z_0).squeeze(-1)            # [B]
    V_loss = (target - V_z).pow(2).mean()

    self.V_optim.zero_grad()
    V_loss.backward()
    self.V_optim.step()

    return V_loss


# ---------------------------------------------------------------------------
# 3. Action-space entropy regulariser (encourages action diversity)
# ---------------------------------------------------------------------------

def action_entropy_loss(self, u_mean, u_std, z_state, num_samples=20, horizon=5):
    """
    Sample latent vectors from the policy distribution, decode them into
    actions, and return the average entropy of the resulting action
    distribution (higher entropy → more diverse actions → reward).

    Args:
        u_mean      (Tensor): [B, latent_dim]
        u_std       (Tensor): [B, latent_dim]
        z_state     (Tensor): [B, z_dim]
        num_samples (int):    Monte-Carlo samples per batch element
        horizon     (int):    planning horizon (used to reshape decoded seq)

    Returns:
        Tensor: scalar entropy (positive; caller negates or uses sign convention)

    Fixes vs. original
    ------------------
    * `agent.cfg.action_shape[0]`  →  `self.cfg.action_dim`
    * decoded_seq is [horizon, B*S, action_dim]; we permute then reshape to
      [B, horizon, S, action_dim] instead of reshaping directly.
    """
    batch      = u_mean.shape[0]
    action_dim = self.cfg.action_dim          # ← was agent.cfg.action_shape[0]

    # Sample latent actions with reparameterisation trick (keeps grad)
    u_dist    = torch.distributions.Normal(u_mean, u_std)
    u_samples = u_dist.rsample((num_samples,))          # [S, B, latent_dim]

    # Flatten S and B for the decoder
    u_flat     = u_samples.reshape(-1, u_samples.shape[-1])   # [S*B, latent_dim]
    z_repeated = z_state.repeat_interleave(num_samples, dim=0) # [S*B, z_dim]

    # decoded_seq: [horizon, S*B, action_dim]
    decoded_seq = self.model.decode_sequence(u_flat, z_repeated)

    # Reshape to [B, horizon, S, action_dim]
    # decoded_seq is [horizon, S*B, action_dim]
    #   permute(1,0,2) → [S*B, horizon, action_dim]
    #   reshape(S, B, horizon, action_dim) → [S, B, horizon, action_dim]
    #   permute(1,2,0,3) → [B, horizon, S, action_dim]
    x = (
        decoded_seq
        .permute(1, 0, 2)                          # [S*B, horizon, action_dim]
        .reshape(num_samples, batch, horizon, action_dim)
        .permute(1, 2, 0, 3)                       # [B, horizon, S, action_dim]
    )

    # Sample covariance across the S dimension → entropy of a Gaussian
    x_mean     = x.mean(dim=2, keepdim=True)       # [B, horizon, 1, action_dim]
    x_centered = x - x_mean                        # [B, horizon, S, action_dim]

    # Empirical covariance: [B, horizon, action_dim, action_dim]
    cov = (1.0 / (num_samples - 1)) * torch.matmul(
        x_centered.transpose(2, 3),   # [B, horizon, action_dim, S]
        x_centered,                   # [B, horizon, S, action_dim]
    )  # → [B, horizon, action_dim, action_dim]

    B, T, _, D = cov.shape
    mean_flat  = x_mean.squeeze(2).reshape(B * T, D)   # [B*T, D]
    cov_flat   = cov.reshape(B * T, D, D)              # [B*T, D, D]

    mvn            = MultivariateNormal(loc=mean_flat, covariance_matrix=cov_flat)
    entropy_flat   = mvn.entropy()              # [B*T]
    entropy        = entropy_flat.view(B, T)    # [B, horizon]

    return entropy.mean(dim=1).mean()           # scalar
