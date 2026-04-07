"""
ο2
----
TDMPC_O2 latent action space extension for TD-MPC.

Public API:
    build_agent  — constructs a TDMPC agent with all Ο2 components attached.

All other symbols can be imported directly from their modules if needed:
    from dcem.action_decoder import build_action_decoder, decode_sequence
    from dcem.planning       import DCEMethod, CEM_in_latent
    from dcem.training       import action_decoder_DDPG_update
"""

from .tdmpc_o2 import TDMPC_O2

__all__ = ["TDMPC_O2"]
