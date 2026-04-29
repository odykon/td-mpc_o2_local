"""
train_o2.py  —  TD-MPC-O2 training entry point.

Run from the project root:

    python train_o2.py task=walker-walk
    python train_o2.py task=cheetah-run exp_name=test latent_action_dim=64

Any key in cfgs/default.yaml or the O2_DEFAULTS dict below can be
overridden via CLI key=value pairs.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import re
import sys
import time
from pathlib import Path

# ── sys.path: make tdmpc/src and project root importable ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TDMPC_SRC    = os.path.join(PROJECT_ROOT, 'tdmpc', 'src')
for _p in (PROJECT_ROOT, TDMPC_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
torch.backends.cudnn.benchmark = True

from omegaconf import OmegaConf

from env import make_env
from algorithm.helper import ReplayBuffer
from implementation.tdmpc_o2 import TDMPC_O2
from implementation.train_pg import train_pg

# ── Paths ─────────────────────────────────────────────────────────────────────
CFGS_DIR    = Path(TDMPC_SRC) / 'cfgs'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ── O2-specific defaults (merged before CLI so CLI always wins) ───────────────
O2_DEFAULTS = {
    # latent action space
    'latent_action_dim': 50,
    'decoder_init':      False,
    'use_latent_state':  True,
    # decoder training
    'dec_reward_coeff':  0.1,
    'variance_schedule': 'linear(0.1, 0.01, 25000)',
    'told_updates':      10,
    # misc
    'exp_name':          'o2',
}


def parse_cfg(cfgs_dir: Path) -> OmegaConf:
    """
    Build config by merging (lowest → highest priority):
      1. cfgs/default.yaml          (base TD-MPC defaults)
      2. cfgs/<modality>.yaml       (pixels overrides, skipped for 'state')
      3. cfgs/tasks/<domain>.yaml   (task-specific values, e.g. action_repeat)
      4. O2_DEFAULTS                (O2 extension defaults)
      5. CLI key=value pairs        (user overrides – highest priority)

    After merging, algebraic string expressions such as "1000/2" that result
    from OmegaConf resolving interpolations like "1000/${action_repeat}" are
    evaluated and replaced with their integer / float values.
    """
    base = OmegaConf.load(cfgs_dir / 'default.yaml')
    cli  = OmegaConf.from_cli()

    # Normalise bare CLI flags (no value → True)
    for k, v in cli.items():
        if v is None:
            cli[k] = True

    # Apply CLI early so task / modality names are available
    base = OmegaConf.merge(base, cli)

    # Modality override
    modality = str(base.get('modality', 'state'))
    if modality not in {'state', 'pixels'}:
        raise ValueError(f'Invalid modality: {modality}')
    if modality != 'state':
        base = OmegaConf.merge(base, OmegaConf.load(cfgs_dir / f'{modality}.yaml'))

    # Task-specific yaml (sets action_repeat etc.)
    task = str(base.task)
    try:
        domain, _ = task.split('-', 1)
    except ValueError:
        raise ValueError(f'Invalid task name "{task}". Expected format: domain-task')
    domain_path = cfgs_dir / 'tasks' / f'{domain}.yaml'
    if not domain_path.exists():
        domain_path = cfgs_dir / 'tasks' / 'default.yaml'
    if domain_path.exists():
        base = OmegaConf.merge(base, OmegaConf.load(domain_path))

    # O2 defaults – merged so that any previously set key (task yaml / CLI) wins
    base = OmegaConf.merge(OmegaConf.create(O2_DEFAULTS), base)

    # Re-apply CLI last so user overrides always win
    base = OmegaConf.merge(base, cli)

    # Resolve algebraic string expressions produced by OmegaConf interpolation
    # e.g. episode_length resolves to "1000/2" → 500
    for k, v in base.items():
        if isinstance(v, str):
            m = re.match(r'(\d+)([+\-*/])(\d+)', v)
            if m:
                result = eval(m.group(1) + m.group(2) + m.group(3))
                base[k] = int(result) if isinstance(result, float) and result.is_integer() else result

    # Convenience fields
    base.task_title = task.replace('-', ' ').title()
    base.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    base.exp_name   = str(base.get('exp_name', 'o2'))

    return base


def main():
    cfg = parse_cfg(CFGS_DIR)

    print(f"\nDevice : {cfg.device}")
    if cfg.device == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    env    = make_env(cfg)
    agent  = TDMPC_O2(cfg)
    buffer = ReplayBuffer(cfg)

    timestamp = time.strftime('%Y-%m-%d_%Hh%M')
    save_dir  = os.path.join(RESULTS_DIR, f"{cfg.exp_name}_{cfg.task}_{timestamp}")

    train_pg(cfg, agent, buffer, env, save_dir=save_dir)


if __name__ == '__main__':
    main()
