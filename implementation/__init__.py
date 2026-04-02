"""
dcem
----
DCEM latent action space extension for TD-MPC.

Public API:
    build_agent  — constructs a TDMPC agent with all DCEM components attached.

All other symbols can be imported directly from their modules if needed:
    from dcem.action_decoder import build_action_decoder, decode_sequence
    from dcem.planning       import DCEMethod, CEM_in_latent
    from dcem.training       import action_decoder_DDPG_update
"""

from dcem.dcem_tdmpc import DCEM_TDMPC

__all__ = ["DCEM_TDMPC"]
