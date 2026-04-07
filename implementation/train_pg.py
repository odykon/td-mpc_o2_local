"""
implementation/train_pg.py

On-policy training loop that uses PGEpisode and the PG decoder update.

Drop-in companion to the existing train.py.  The only structural difference
from the original `train()` is:
  • Uses PGEpisode instead of Episode (adds on-policy fields automatically)
  • Calls update_decoder_pg() instead of the off-policy decoder update
  • Keeps the exact same TOLD model-update block as the original
"""

import os
import time

import torch
import pandas as pd
from copy import deepcopy

import helper as h

# Adjust these imports to match your project's actual module paths
import lml as l
from implementation.episode import PGEpisode   # ← subclass, not a replacement


# ---------------------------------------------------------------------------
# Helper: one epoch of on-policy decoder updates for a single episode
# ---------------------------------------------------------------------------

def update_decoder_pg(agent, episode: PGEpisode, step: int) -> dict:
    """
    Run one full epoch of mini-batch PG updates over the transitions stored
    in *episode*.

    The function:
      1. Freezes TOLD model gradients
      2. Unfreezes decoder + value-network parameters
      3. Iterates over the episode in shuffled mini-batches
      4. Re-enables TOLD model gradients when done

    Args:
        agent:   DCEM_TDMPC (or any agent with .PG_withV, .DCEMethod, .cfg)
        episode: PGEpisode that has already been finalized
        step:    current global environment step (for schedules)

    Returns:
        dict[str, float]: mean of each loss component across all mini-batches
    """
    agent.model.track_TOLD_grad(False)
    for m in [agent.model._action_decoder, agent.model._V]:
        h.set_requires_grad(m, True)

    alpha_v   = h.linear_schedule(agent.cfg.variance_schedule, step)
    horizon   = int(h.linear_schedule(agent.cfg.horizon_schedule, step))
    accum: dict[str, list] = {}

    for batch in episode.sample_batches(batch_size=128, shuffle=True):
        (obs_t, action_t, reward_t, obs_t1,
         _old_log, latent_action_t, _umean, _ustd,
         _done, next_rewards, next_obses) = batch

        # Re-run CEM to get fresh distribution parameters (on-policy)
        _, u_mean, u_std, _, _ = agent.DCEMethod(
            obs_t, update_mode=True, step=step, t0=False
        )

        loss_dict = agent.PG_withV(
            obs_t, u_mean, u_std, reward_t, obs_t1,
            latent_action_t, next_rewards, next_obses, alpha_v, horizon,
        )

        for k, v in loss_dict.items():
            accum.setdefault(k, []).append(v)

    for m in [agent.model._action_decoder, agent.model._V]:
        h.set_requires_grad(m, False)
    agent.model.track_TOLD_grad(True)

    return {k: torch.stack(v).mean().item() for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Main on-policy training loop
# ---------------------------------------------------------------------------

def train_pg(cfg, agent, buffer, env, save_dir=None):
    """
    Training loop that uses PGEpisode and on-policy decoder updates.

    Structurally identical to the original train() except for:
      • PGEpisode with episode.add_pg(…) to store on-policy data
      • update_decoder_pg() for the decoder update step

    Args:
        cfg:      Hydra / OmegaConf config object
        agent:    DCEM_TDMPC instance (must have PG_withV, DCEMethod, …)
        buffer:   ReplayBuffer
        env:      Gym-compatible environment
        save_dir: (optional) path for results; auto-created if None
    """
    h.set_seed(42)
    episode_idx  = 0
    start_time   = time.time()

    if save_dir is None:
        save_dir = l.make_save_dir_path(agent.cfg, base_dir="results")
    print(f"Saving results to: {save_dir}")

    # ---- Config summary ---------------------------------------------------
    print("\n" + "=" * 50)
    print("\033[1m🚀 Training Configuration (PG mode)\033[0m")
    print("=" * 50)
    for key, value in agent.cfg.items():
        print(f"  {key:30}: {value}")
    print("=" * 50 + "\n")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for step in range(0, cfg.train_steps, cfg.episode_length):

        obs     = env.reset()
        episode = PGEpisode(cfg, obs)       # ← PGEpisode, not Episode
        current_step     = 0
        half_time_reward = 0.0

        # ---- Rollout -------------------------------------------------------
        while not episode.done:
            if step < cfg.seed_steps:
                action_np = env.action_space.sample()
                action    = torch.tensor(action_np, dtype=torch.float32,
                                         device=agent.device)
            else:
                action, u_mean, u_std, latent_action, log_prob = (
                    agent.CEM_in_latent(
                        obs, step=step, t0=episode.first,
                        seed=None, sample_final_action=True,
                    )
                )
                # Store on-policy data for this step
                episode.add_pg(log_prob, u_mean, u_std, latent_action)

            obs, reward, done, _ = env.step(action.cpu().numpy())
            current_step        += 1
            if current_step <= 500:
                half_time_reward += reward

            episode += (obs, action, reward, done)

        # ---- Episode summary -----------------------------------------------
        episode_metrics = {
            "Episode_no":       int(step / cfg.episode_length),
            "Reward":           episode.cumulative_reward,
            "Half_time_reward": half_time_reward,
            "Horizon":          int(h.linear_schedule(cfg.horizon_schedule, step)),
            "Std":              h.linear_schedule(cfg.std_schedule, step),
        }
        print("\n  Episode Summary")
        print("-" * 25)
        for k, v in episode_metrics.items():
            print(f"  {k:20}: {v}")

        # ---- Add episode to replay buffer ----------------------------------
        buffer += episode
        print(f"  Buffer idx: {buffer.idx}, full: {buffer._full}")

        train_metrics   = {}
        decoder_metrics = {}

        # ---- On-policy decoder update (only after seed phase) -------------
        if step >= cfg.seed_steps:
            t0 = time.time()
            decoder_metrics = update_decoder_pg(agent, episode, step)
            train_metrics["decoder_update_s"] = time.time() - t0

            print("  Decoder Update Metrics:")
            for k, v in decoder_metrics.items():
                print(f"    {k:20}: {v:.4f}")

        # ---- TOLD model update (off-policy from buffer) -------------------
        if step >= cfg.seed_steps:
            buffer.cfg.batch_size = 512
            num_updates = agent.cfg.told_updates
            told_accum: dict[str, float] = {}
            for i in range(num_updates):
                m = agent.update(buffer, step + i)
                for k, v in m.items():
                    told_accum[k] = told_accum.get(k, 0.0) + v
            for k in told_accum:
                told_accum[k] /= num_updates
            train_metrics.update(told_accum)

        print("  Training Metrics:")
        for k, v in train_metrics.items():
            print(f"    {k:20}: {v:.4f}" if isinstance(v, float) else
                  f"    {k:20}: {v}")

        # ---- Logging -------------------------------------------------------
        if step >= cfg.seed_steps:
            full_metrics = {**episode_metrics, **train_metrics, **decoder_metrics}
            l.save_results(agent.cfg, full_metrics, save_dir, None, step)

        episode_idx += 1

    # ---- Final evaluation --------------------------------------------------
    final_eval = l.evaluate_agent(
        env, agent, agent.cfg, step,
        cem=False, LML=False, n_episodes=2,
        save_dir=save_dir, video_mode="none",
    )
    path = os.path.join(save_dir, "final_eval.csv")
    df   = pd.DataFrame([final_eval])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    l.save_model_and_buffer(agent, buffer, save_dir)
