"""
planning.py
-----------
Planning methods for latent action space control.

Provides two functions that are exposed as methods on DCEM_TDMPC:

    DCEMethod     — Differentiable CEM using the LML soft top-k projection.
                    When update_mode=True, gradients flow through the elite
                    selection. Used for decoder training (see training.py).

    CEM_in_latent — Vanilla CEM with hard top-k elite selection.
                    Non-differentiable. Used for environment interaction
                    during training and as an ablation baseline.

Both functions take self as their first argument where self is a
DCEM_TDMPC instance, giving access to self.model, self.cfg,
self.device, and self.estimate_value.

Dependencies
------------
    lml — Limited Multi-Label Projection (Amos et al., 2019)
          pip install git+https://github.com/locuslab/lml.git
"""

import torch

import algorithm.helper as h
from lml import LML

def DCEMethod(self, obs, update_mode=False, step=None, t0=True,
              seed=None, sample_final_action=False, lml_temperature=10):
    """
    Plan using the Differentiable Cross-Entropy Method in latent action space.

    Args:
        obs:                  Raw observation. Converted to tensor internally
                              when update_mode=False.
        update_mode:          If True, retain computation graph for decoder
                              training. Runs under torch.enable_grad().
                              If False, runs under torch.no_grad().
        step:                 Current training step (for horizon schedule).
        t0:                   Whether this is the first step of an episode.
        seed:                 Optional int seed for reproducible noise sampling.
        sample_final_action:  If True, sample from the final distribution
                              instead of using the mean.
        lml_temperature:      Temperature for the LML soft top-k projection.
                              Higher -> sharper elite selection.

    Returns:
        action:        [action_dim]           first action of planned sequence.
        u_mean:        [B, latent_action_dim] final search distribution mean.
        u_std:         [B, latent_action_dim] final search distribution std.
        latent_action: [B, latent_action_dim] latent action that was decoded.
        log_probs:     scalar                 log prob of latent_action.
    """
    if not update_mode:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    context = torch.enable_grad if update_mode else torch.no_grad
    with context():
        # Encode and tile latent state
        z = self.model.h(obs)                                              # [B, latent_dim]
        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
        z = z.view(B * self.cfg.num_samples, -1)                           # [B*N, latent_dim]
        if update_mode:
            z = z.detach()

        # Initialise search distribution
        u_mean = torch.zeros(B, self.cfg.latent_action_dim,
                             device=self.cfg.device, requires_grad=update_mode)
        u_std  = 2 * torch.ones(B, self.cfg.latent_action_dim,
                                device=self.cfg.device, requires_grad=update_mode)

        # Optional per-element RNG for reproducibility
        if seed is not None:
            gens = h.sample_u_noise_generators(B, self.cfg.device, seed)

        # CEM iterations
        for i in range(self.cfg.iterations):
            if seed is not None:
                noises = [
                    torch.randn(1, self.cfg.num_samples, self.cfg.latent_action_dim,
                                device=self.cfg.device, generator=gens[b])
                    for b in range(B)
                ]
                u_noise = torch.cat(noises, dim=0)                         # [B, N, d_u]
            else:
                u_noise = torch.randn(B, self.cfg.num_samples,
                                      self.cfg.latent_action_dim, device=self.cfg.device)

            u_samples      = u_mean.unsqueeze(1) + u_std.unsqueeze(1) * u_noise  # [B, N, d_u]
            u_samples_flat = u_samples.view(B * self.cfg.num_samples, self.cfg.latent_action_dim)

            sequence = self.model.decode_sequence(u_samples_flat, z)
            value    = self.estimate_value_with_grad(z, sequence, horizon).view(B, self.cfg.num_samples)

            value_mean   = value.mean(dim=1, keepdim=True)
            value_std    = value.std(dim=1, keepdim=True) + 1e-8
            value_normed = (value - value_mean) / value_std

            # LML soft top-k elite selection
            scores = LML(N=self.cfg.num_elites, verbose=0, eps=1e-4)(value_normed * lml_temperature)
            scores = scores / scores.sum(dim=1, keepdim=True)              # [B, N]

            w   = scores.unsqueeze(2)                                      # [B, N, 1]
            u_m = (w * u_samples).sum(dim=1)                               # [B, d_u]
            u_s = torch.sqrt(
                (w * (u_samples - u_m.unsqueeze(1)) ** 2).sum(dim=1)
                / (scores.sum(dim=1, keepdim=True) + 1e-9)
            ).clamp(self.std, 2)

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s

        # Decode final action
        z_0   = self.model.h(obs).detach()
        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        log_probs     = dist.log_prob(latent_action).squeeze_(0).sum(dim=0)

        sequence = self.model.decode_sequence(latent_action, z_0)
        action   = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs


def CEM_in_latent(self, obs, update_mode=False, step=None, t0=True,
                  seed=None, sample_final_action=False, lml_temperature=10):
    """
    Plan using vanilla (non-differentiable) CEM in latent action space.

    Identical interface to DCEMethod. Uses hard top-k elite selection
    instead of LML. Always runs under torch.no_grad().

    Use for environment interaction during training and as the ablation
    baseline in experiments.

    Args:
        obs:                 Raw observation.
        step:                Current training step (for horizon schedule).
        sample_final_action: Sample from final distribution instead of mean.

    Returns:
        Same tuple as DCEMethod.
    """
    if not update_mode:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
    B = obs.shape[0]
    horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

    with torch.no_grad():
        # Encode and tile latent state
        z = self.model.h(obs)
        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
        z = z.view(B * self.cfg.num_samples, -1)

        # Initialise search distribution
        u_mean = torch.zeros(self.cfg.latent_action_dim, device=self.cfg.device)
        u_std  = 2 * torch.ones(self.cfg.latent_action_dim, device=self.cfg.device)

        # CEM iterations
        for i in range(self.cfg.iterations):
            u_noise   = torch.randn(self.cfg.num_samples, self.cfg.latent_action_dim,
                                    device=self.cfg.device)
            u_samples = u_mean.unsqueeze(0) + u_std.unsqueeze(0) * u_noise  # [N, d_u]

            sequence = self.model.decode_sequence(u_samples, z)
            value    = self.estimate_value(z, sequence, horizon).squeeze(1)  # [N]

            # Hard top-k elite selection
            elite_idxs    = torch.topk(value, self.cfg.num_elites, dim=0).indices
            elite_samples = u_samples[elite_idxs]

            u_m = elite_samples.mean(dim=0)
            u_s = elite_samples.std(dim=0, unbiased=False).clamp(self.std, 2)

            u_mean = self.cfg.momentum * u_mean + (1 - self.cfg.momentum) * u_m
            u_std  = u_s

        # Decode final action
        z_0   = self.model.h(obs)
        dist  = torch.distributions.Normal(loc=u_mean, scale=u_std)
        latent_action = dist.rsample() if sample_final_action else u_mean
        latent_action = latent_action.unsqueeze(0)

        log_probs = dist.log_prob(latent_action).squeeze_(0).sum(dim=0)
        sequence  = self.model.decode_sequence(latent_action, z_0)
        action    = sequence[0, :].squeeze_(0)

    return action, u_mean, u_std, latent_action, log_probs
