"""
dcem_tdmpc.py
-------------
DCEM_TDMPC: subclass of TDMPC that adds the DCEM latent action space extension.

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
from implementation.action_decoder import build_action_decoder, decode_sequence,track_TOLD_grad
from implementation.planning import DCEMethod, CEM_in_latent
from implementation.training import action_decoder_DDPG_update

class DCEM_TDMPC(TDMPC):
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

        # Attach decode_sequence to model and model_target
        self.model.decode_sequence        = types.MethodType(decode_sequence, self.model)
        self.model_target.decode_sequence = types.MethodType(decode_sequence, self.model_target)
        self.model.track_TOLD_grad        = types.MethodType(track_TOLD_grad, self.model)
        self.model_target.track_TOLD_grad = types.MethodType(track_TOLD_grad, self.model_target)

    def DCEMethod(self, *args, **kwargs):
        return DCEMethod(self, *args, **kwargs)

    def CEM_in_latent(self, *args, **kwargs):
        return CEM_in_latent(self, *args, **kwargs)

    def action_decoder_DDPG_update(self, *args, **kwargs):
        return action_decoder_DDPG_update(self, *args, **kwargs)
