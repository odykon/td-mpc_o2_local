"""
tdmpc_o2.py
-------------
TDMPC_O2: subclass of TDMPC that adds the DCEM latent action space extension.

Inherits all TD-MPC functionality (update, plan, save, load, etc.) unchanged.
Adds in __init__:
    - action decoder attached to self.model and self.model_target
    - decode_sequence method attached to self.model and self.model_target
    - action_dec_optim for decoder training

Adds three methods:
    - DCEMethod                  — differentiable CEM planner
    - CEM_in_latent              — vanilla CEM planner (ablation baseline)
    - action_decoder_DDPG_update — DDPG-style decoder update
"""

import types
import torch
from copy import deepcopy

from algorithm.tdmpc import TDMPC, TOLD
from implementation.action_decoder import build_action_decoder, decode_sequence, track_TOLD_grad, track_O2_grad,build_value_network
from implementation.planning import DCEMethod, CEM_in_latent
from implementation.training import action_decoder_DDPG_update, PG_withV, action_entropy_loss, V_net_update

class TDMPC_O2(TDMPC):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Build decoder and attach to model and model_target
        decoder = build_action_decoder(
            cfg,
            initialize=cfg.decoder_init,
            use_latent_state=cfg.use_latent_state,
        ).to(self.device)


        self.model._action_decoder        = decoder
        self.model_target._action_decoder = deepcopy(decoder).to(self.device)

        # Decoder optimiser
        self.action_dec_optim = torch.optim.Adam(
            self.model._action_decoder.parameters(),
            lr=cfg.lr,
        )

        #attach value network to model and model_target
        self.model._V        = build_value_network(cfg.latent_dim, cfg.mlp_dim).to(self.device)
        self.model_target._V = deepcopy(self.model._V).to(self.device)
        self.V_optim = torch.optim.Adam(self.model._V.parameters(), lr=cfg.lr)

        # Attach decode_sequence to model and model_target
        self.model.decode_sequence        = types.MethodType(decode_sequence, self.model)
        self.model_target.decode_sequence = types.MethodType(decode_sequence, self.model_target)
        self.model.track_TOLD_grad        = types.MethodType(track_TOLD_grad, self.model)
        self.model_target.track_TOLD_grad = types.MethodType(track_TOLD_grad, self.model_target)
        self.model.track_O2_grad          = types.MethodType(track_O2_grad, self.model)
        self.model_target.track_O2_grad   = types.MethodType(track_O2_grad, self.model_target)

    def estimate_value_with_grad(self, z, actions, horizon):
        """estimate_value without @torch.no_grad() for gradient flow in DCEMethod."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G
  
    def DCEMethod(self, *args, **kwargs):
        return DCEMethod(self, *args, **kwargs)

    def CEM_in_latent(self, *args, **kwargs):
        return CEM_in_latent(self, *args, **kwargs)

    def action_decoder_DDPG_update(self, *args, **kwargs):
        return action_decoder_DDPG_update(self, *args, **kwargs)
        
    def PG_withV(self, *args, **kwargs):
        return PG_withV(self, *args, **kwargs)
    
    def action_entropy_loss(self, *args, **kwargs):
        return action_entropy_loss(self, *args, **kwargs)
    
    def V_net_update(self, *args, **kwargs):
        return V_net_update(self, *args, **kwargs)
