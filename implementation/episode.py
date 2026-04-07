"""
implementation/episode.py

PGEpisode extends the original TDMPC Episode with on-policy fields needed
for policy-gradient decoder updates (u_mean, u_std, latent_action, log_probs).

Usage
-----
Replace every `Episode(cfg, obs)` call in the PG training loop with
`PGEpisode(cfg, obs)`.  The original `Episode` class is unchanged, so
all existing replay-buffer code keeps working without modification.
"""

import torch

# Import the original Episode from wherever it lives in your repo.
# Adjust the path if your project lays it out differently.
from tdmpc.episode import Episode   # ← original TDMPC Episode


class PGEpisode(Episode):
    """
    Superset of the original TDMPC Episode that additionally stores
    the on-policy information produced by CEM_in_latent:

        old_log_probs   – scalar log-probability of the chosen latent action
        u_mean          – mean of the latent Normal distribution  [latent_dim]
        u_std           – std  of the latent Normal distribution  [latent_dim]
        latent_action   – the sampled latent action               [latent_dim]

    All four lists are populated step-by-step during rollout (just like
    ``self.action``), then converted to tensors in ``finalize()``.
    ``sample_batches`` yields them alongside the standard fields.
    """

    def __init__(self, cfg, init_obs):
        super().__init__(cfg, init_obs)

        # On-policy extras – one entry per environment step
        self.old_log_probs: list = []
        self.u_mean:        list = []
        self.u_std:         list = []
        self.latent_action: list = []

    # ------------------------------------------------------------------
    # Public helper called from the training loop
    # ------------------------------------------------------------------

    def add_pg(self, log_prob, u_mean, u_std, latent_action):
        """
        Store the on-policy tensors produced by CEM_in_latent for the
        current time-step.  Call this *before* ``episode + (obs, action,
        reward, done)``, or at least before ``finalize()``.

        Args:
            log_prob       (Tensor): scalar or shape [] – log prob of chosen action
            u_mean         (Tensor): [latent_dim]
            u_std          (Tensor): [latent_dim]
            latent_action  (Tensor): [latent_dim]
        """
        self.old_log_probs.append(log_prob)
        self.u_mean.append(u_mean.squeeze(0))
        self.u_std.append(u_std.squeeze(0))
        self.latent_action.append(latent_action.squeeze(0))

    # ------------------------------------------------------------------
    # Override finalize to also stack the new fields
    # ------------------------------------------------------------------

    def finalize(self):
        """Finalize the base episode, then stack on-policy tensors."""
        super().finalize()

        if len(self.old_log_probs) > 0:
            self.old_log_probs  = torch.stack(self.old_log_probs)
            self.u_mean         = torch.stack(self.u_mean)
            self.u_std          = torch.stack(self.u_std)
            self.latent_action  = torch.stack(self.latent_action)
        # else: lists remain empty – episode was never updated with PG data

    # ------------------------------------------------------------------
    # Override sample_batches to yield the extra fields
    # ------------------------------------------------------------------

    def sample_batches(self, batch_size=None, shuffle=False):
        """
        Iterate through all transitions in batches.

        Yields 11-tuples:
            obs_t, action_t, reward_t, obs_t1,
            old_log_probs_t, latent_action_t, u_mean_t, u_std_t,
            done_t,
            next_rewards,   # [B, 5]
            next_obs        # [B, 5, *obs_shape]
        """
        if not isinstance(self.obs, torch.Tensor):
            raise RuntimeError(
                "Episode must be finalized before sampling. "
                "Call finalize() or wait for the episode to end."
            )

        episode_length = len(self)
        obs_shape = self.obs.shape[1:]

        obs_t  = self.obs[:-1]
        obs_t1 = self.obs[1:]

        done_t = torch.zeros(episode_length, dtype=torch.bool, device=self.device)
        if self.done:
            done_t[-1] = True

        # ---- next-5-rewards / next-5-obs (same logic as original) ------
        next_rewards = torch.zeros((episode_length, 5),
                                   dtype=torch.float32, device=self.device)
        dtype_obs    = self.obs.dtype
        next_obs     = torch.zeros((episode_length, 5) + obs_shape,
                                   dtype=dtype_obs, device=self.device)

        for t in range(episode_length):
            avail_r = min(5, episode_length - t)
            if avail_r > 0:
                next_rewards[t, :avail_r] = self.reward[t:t + avail_r]
            avail_o = min(5, episode_length - t)
            if avail_o > 0:
                next_obs[t, :avail_o] = self.obs[t + 1:t + 1 + avail_o]

        # ---- batching ---------------------------------------------------
        indices = torch.arange(episode_length, device=self.device)
        if shuffle:
            indices = indices[torch.randperm(episode_length)]

        if batch_size is None:
            batch_size = episode_length

        for start in range(0, episode_length, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                obs_t[idx],
                self.action[idx],
                self.reward[idx],
                obs_t1[idx],
                self.old_log_probs[idx],
                self.latent_action[idx],
                self.u_mean[idx],
                self.u_std[idx],
                done_t[idx],
                next_rewards[idx],
                next_obs[idx],
            )
