"""
training_utils.py
-----------------
Shared training utilities used across all training scripts.

Functions:
    collect_episode      — roll out one episode using CEM_in_latent
    update_tdmpc         — update the TOLD world model (works with TDMPC and TDMPC_O2)
    update_decoder       — DDPG-style decoder update loop
    update_decoder_pg    — on-policy PG decoder update over a PGEpisode
"""

import random
import torch
import numpy as np
from algorithm.helper import Episode, linear_schedule
from o2.episode import PGEpisode


def sample_recent_obs(buffer, n):
    """
    Sample a batch of observations from the n most recent transitions in the buffer.

    Uses PER weights computed over the recent window only (not the full buffer),
    which avoids the weighting bug in the original sample_new implementation.

    Args:
        buffer: ReplayBuffer instance
        n:      number of most recent transitions to sample from

    Returns:
        obs: [batch_size, obs_dim] observation tensor
    """
    total = int(buffer._full) * buffer.capacity + (not buffer._full) * buffer.idx
    n = min(n, total)
    end   = buffer.idx
    start = (end - n) % buffer.capacity

    if start < end:
        recent_priorities = buffer._priorities[start:end]
        offset = start
    else:
        recent_priorities = torch.cat([buffer._priorities[start:], buffer._priorities[:end]])
        offset = start

    probs = recent_priorities ** buffer.cfg.per_alpha
    probs = probs / probs.sum()

    rel_idxs = torch.from_numpy(
        np.random.choice(len(probs), buffer.cfg.batch_size, p=probs.cpu().numpy(), replace=False)
    ).to(buffer.device)

    idxs = (rel_idxs + offset) % buffer.capacity
    return buffer._get_obs(buffer._obs, idxs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_episode(env, agent, cfg, step):
    """
    Roll out one episode using CEM_in_latent planning.

    During seed steps, actions are sampled uniformly from the action space.
    After seed steps, CEM_in_latent is used. If the agent is a TDMPC_O2
    instance, on-policy data (log_probs, u_mean, u_std, latent_action) is
    stored via episode.add_pg().

    Args:
        env:   Gym-compatible environment
        agent: TDMPC or TDMPC_O2 instance
        cfg:   OmegaConf config
        step:  current global step (for seed_steps check and planning schedules)

    Returns:
        Episode or PGEpisode
    """
    obs = env.reset()
    is_o2 = hasattr(agent, 'CEM_in_latent')
    episode = PGEpisode(cfg, obs) if is_o2 else Episode(cfg, obs)

    while not episode.done:
        if step < cfg.seed_steps:
            action_np = env.action_space.sample()
            action = torch.tensor(action_np, dtype=torch.float32, device=agent.device)
        else:
            if is_o2:
                action, u_mean, u_std, latent_action, log_prob = agent.CEM_in_latent(
                    obs, step=step, t0=episode.first, seed=None, sample_final_action=True
                )
                episode.add_pg(log_prob, u_mean, u_std, latent_action)
            else:
                action = agent.plan(obs, step=step, t0=episode.first)

        obs, reward, done, _ = env.step(action.cpu().numpy())
        episode += (obs, action, reward, done)

    if is_o2:
        episode.finalize()

    return episode


def update_tdmpc(agent, buffer, step):
    """
    Update the TOLD world model for cfg.told_updates iterations.

    Works with both TDMPC and TDMPC_O2. When used with TDMPC_O2, ensures
    TOLD gradients are enabled and decoder gradients are disabled for the
    duration of the update.

    Args:
        agent:  TDMPC or TDMPC_O2 instance
        buffer: ReplayBuffer
        step:   current global step

    Returns:
        dict of mean loss metrics across all update iterations
    """
    if hasattr(agent.model, 'track_TOLD_grad'):
        agent.model.track_TOLD_grad(True)
        agent.model.track_O2_grad(False)

    buffer.cfg.batch_size = agent.cfg.batch_size
    num_updates = getattr(agent.cfg, 'told_updates', agent.cfg.episode_length)
    metrics = {}
    for i in range(num_updates):
        update_metrics = agent.update(buffer, step + i)
        for k, v in update_metrics.items():
            metrics[k] = metrics.get(k, 0.0) + v
    for k in metrics:
        metrics[k] /= agent.cfg.told_updates

    if hasattr(agent.model, 'track_O2_grad'):
        agent.model.track_O2_grad(True)

    return metrics


def update_decoder(agent, buffer, cfg, step):
    """
    DDPG-style decoder update loop (off-policy).

    Freezes TOLD, samples batches from the buffer, runs DCEMethod to get
    differentiable latent action means, then calls action_decoder_DDPG_update.

    Args:
        agent:  TDMPC_O2 instance
        buffer: ReplayBuffer
        cfg:    OmegaConf config
        step:   current global step

    Returns:
        float: mean decoder loss across all update iterations
    """
    agent.model.track_TOLD_grad(False)
    buffer.cfg.batch_size = agent.cfg.dcem_batch_size
    horizon = int(linear_schedule(cfg.horizon_schedule, step))

    n = getattr(agent.cfg, 'dcem_sampling_n', None)
    total_loss = 0.0
    total_grad_norm = 0.0
    for _ in range(agent.cfg.decoder_updates):
        obs = sample_recent_obs(buffer, n) if n else buffer.sample()[0]
        _, u_mean, _, _, _ = agent.DCEMethod(obs, update_mode=True, step=step, t0=False)
        loss, grad_norm = agent.action_decoder_DDPG_update(obs, u_mean, horizon)
        total_loss += loss
        total_grad_norm += grad_norm

    agent.model.track_TOLD_grad(True)
    n_updates = agent.cfg.decoder_updates
    return total_loss / n_updates, total_grad_norm / n_updates


def update_decoder_pg(agent, episode, step):
    """
    On-policy PG decoder update over a full PGEpisode.

    Freezes TOLD, unfreezes decoder and value network, iterates over the
    episode in shuffled mini-batches, re-runs DCEMethod to get fresh
    distribution parameters, then calls PG_withV.

    Args:
        agent:   TDMPC_O2 instance
        episode: PGEpisode (already finalized)
        step:    current global step

    Returns:
        dict of mean loss metrics across all mini-batches
    """
    agent.model.track_TOLD_grad(False)
    agent.model.track_O2_grad(True)

    horizon = int(linear_schedule(agent.cfg.horizon_schedule, step))
    accum = {}

    for batch in episode.sample_batches(batch_size=agent.cfg.dcem_batch_size, shuffle=True):
        (obs_t, action_t, reward_t, obs_t1,
         _old_log, latent_action_t, _umean, _ustd,
         _done, next_rewards, next_obses) = batch

        _, u_mean, u_std, _, _ = agent.DCEMethod(
            obs_t, update_mode=True, step=step, t0=False
        )

        loss_dict = agent.PG_withV(
            obs_t, u_mean, u_std, reward_t, obs_t1,
            latent_action_t, next_rewards, next_obses, horizon, _old_log
        )

        for k, v in loss_dict.items():
            accum.setdefault(k, []).append(v)

    agent.model.track_O2_grad(False)
    agent.model.track_TOLD_grad(True)

    return {k: sum(v) / len(v) for k, v in accum.items()}
