"""
ppo_training.py
---------------
PPO-style decoder update for the latent action space.
 
Provides:
    action_decoder_PPO — Proximal Policy Optimisation style update for the
                         action decoder, exposed as a method on TDMPC_O2.
 
This implements a KL-penalised policy gradient update (PPO with KL penalty
rather than clipping) for the action decoder. The decoder is treated as a
policy in the latent action space and updated to maximise advantage while
staying close to the behaviour policy that collected the data.
 
Two variants are implemented and can be selected via the `use_kl` flag:
    use_kl=False  — plain policy gradient (REINFORCE-style)
    use_kl=True   — KL-penalised PG (PPO-KL variant)
 
The clipping variant (PPO-clip) is included as commented code for reference.
 
Advantage estimation
--------------------
Uses a Monte Carlo estimate of the return bootstrapped with the world model's
Q-function at the terminal state:
 
    A(s, u) = Σ_t γ^t r_t + γ^T * Q(z_T, π(z_T)) - V(z_0)
 
Where V(z_0) is estimated by averaging Q-values over samples from the
current latent distribution (calculate_baselines).
 
Reference
---------
    Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
    https://arxiv.org/abs/1707.06347
"""
 
import torch
import torch.nn as nn
 
 
def action_decoder_PPO(
    self,
    obs,
    u_mean,
    u_std,
    reward,
    next_obses,
    original_log_probs,
    original_action,
    original_u_mean,
    original_u_std,
    use_kl=False,
    beta=0.05,
):
    """
    PPO-style decoder update step.
 
    'self' refers to the TDMPC_O2 instance, giving access to self.model,
    self.estimate_value_with_grad, self.action_dec_optim, and self.cfg.
 
    Args:
        obs:                [B, obs_dim] current observations.
        u_mean:             [B, latent_action_dim] current distribution mean
                            from DCEMethod(update_mode=True).
        u_std:              [B, latent_action_dim] current distribution std.
        reward:             [T, B] or [T] rewards collected over T steps.
        next_obses:         [T, B, obs_dim] next observations. The last
                            element is used for terminal value bootstrapping.
        original_log_probs: [B] log probabilities under the behaviour policy.
        original_action:    [B, latent_action_dim] latent actions that were
                            executed during data collection.
        original_u_mean:    [B, latent_action_dim] behaviour policy mean.
        original_u_std:     [B, latent_action_dim] behaviour policy std.
        use_kl:             If True, add KL penalty between current and
                            behaviour policy (PPO-KL). If False, plain PG.
        beta:               Coefficient for the KL penalty term.
 
    Returns:
        Scalar loss value for logging.
    """
    gamma = self.cfg.discount
 
    # ---- Monte Carlo return estimate ----
    discounted_rewards = 0.0
    current_discount   = 1.0
    for t in range(reward.shape[0]):
        discounted_rewards += current_discount * reward[t]
        current_discount   *= gamma
 
    # Bootstrap terminal value with Q-function
    with torch.no_grad():
        z_final  = self.model.h(next_obses[-1])
        Q_final  = self.calculate_baselines(z_final, u_mean, u_std, max=True)
        mc_return = (discounted_rewards + current_discount * Q_final).squeeze(-1)
 
    # ---- Advantage ----
    with torch.no_grad():
        z_0      = self.model.h(obs)
        baseline = self.calculate_baselines(z_0, u_mean, u_std, max=True).squeeze(-1)
 
    advantage = mc_return - baseline
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
 
    # ---- Log probability ratio ----
    dist_new = torch.distributions.Normal(loc=u_mean, scale=u_std)
    dist_old = torch.distributions.Normal(loc=original_u_mean, scale=original_u_std)
 
    # Per-dimension mean to keep scale independent of latent_action_dim
    current_log_probs = dist_new.log_prob(original_action).mean(dim=-1)
    old_log_probs     = dist_old.log_prob(original_action).mean(dim=-1)
 
    # ---- Policy gradient loss ----
    PG_loss = current_log_probs * advantage
 
    # ---- Optional KL penalty ----
    if use_kl:
        kl   = torch.distributions.kl.kl_divergence(dist_old, dist_new).sum(dim=-1)
        loss = -(PG_loss - beta * kl).mean()
    else:
        loss = -PG_loss.mean()
 
    # PPO-clip variant (for reference):
    # ratio     = torch.exp(current_log_probs - old_log_probs)
    # clip_ratio = 0.2
    # loss = -torch.min(
    #     ratio * advantage,
    #     torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
    # ).mean()
 
    # ---- Update ----
    self.action_dec_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        self.model._action_decoder.parameters(), max_norm=1
    )
    self.action_dec_optim.step()
 
    return loss.item()
 
