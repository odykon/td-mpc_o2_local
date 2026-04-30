"""
TD-MPC O2 training script — DDPG decoder update.

Training proceeds in two phases:
  Phase 1 (step < decoder_start_steps):
      - Planning with standard CEM (agent.plan)
      - Only TOLD world model is updated

  Phase 2 (step >= decoder_start_steps):
      - Planning switches to CEM in latent space (agent.CEM_in_latent)
      - TOLD updated first (TOLD grad on, decoder grad off)
      - Decoder updated after (decoder grad on, TOLD grad off)

Usage (from repo root):
    python scripts/train_o2_ddpg.py task=walker-walk seed=1
    python scripts/train_o2_ddpg.py cfg=my_cfg.yaml

Kaggle setup:
    !git clone <your-repo-url>
    %cd <repo-name>
    !pip install dm-control omegaconf
    !python scripts/train_o2_ddpg.py task=walker-walk

Logs saved to logs/<task>/<modality>/<exp_name>/<seed>/
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tdmpc' / 'src'))

import torch
import numpy as np
import random
import time
import csv

from omegaconf import OmegaConf
from cfg import parse_cfg
from env import make_env
from algorithm.helper import Episode, ReplayBuffer, linear_schedule
from o2.tdmpc_o2 import TDMPC_O2
from o2.training_utils import update_tdmpc, update_decoder

torch.backends.cudnn.benchmark = True

CFG_PATH = REPO_ROOT / 'tdmpc' / 'cfgs'
LOG_ROOT  = REPO_ROOT / 'logs'

O2_DEFAULTS = {
    'latent_action_dim':    128,
    'decoder_init':         True,
    'use_latent_state':     True,
    'dcem_batch_size':      64,
    'decoder_updates':      50,
    'told_updates':         500,
    'decoder_start_steps':  5000,
    'exp_name':             'o2_ddpg',
    # CEM hyperparams for O2 latent-space planning (independent of standard CEM)
    'latent_num_samples':   32,
    'latent_num_elites':    8,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(env, agent, num_episodes: int, step: int) -> float:
    """Run agent in eval mode using CEM in latent space."""
    rewards = []
    for _ in range(num_episodes):
        obs, done, total, t = env.reset(), False, 0.0, 0
        while not done:
            action, *_ = agent.CEM_in_latent(obs, step=step, t0=(t == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            total += reward
            t += 1
        rewards.append(total)
    return float(np.mean(rewards))


class CSVLogger:
    """Logs train and eval metrics to separate CSV files with console output."""

    TRAIN_FIELDS = [
        'episode', 'step', 'env_step', 'total_time', 'episode_reward',
        'horizon', 'std', 'ep_time', 'update_time', 'decoder_time',
        'decoder_loss', 'phase',
        'consistency_loss', 'reward_loss', 'value_loss',
        'pi_loss', 'total_loss', 'weighted_loss', 'grad_norm',
    ]
    EVAL_FIELDS = ['episode', 'env_step', 'episode_reward', 'total_time']

    def __init__(self, log_dir: Path, cfg):
        log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, log_dir / 'config.yaml')

        self._train_f = open(log_dir / 'train.csv', 'w', newline='')
        self._eval_f  = open(log_dir / 'eval.csv',  'w', newline='')
        self._train_w = csv.DictWriter(self._train_f, fieldnames=self.TRAIN_FIELDS, extrasaction='ignore')
        self._eval_w  = csv.DictWriter(self._eval_f,  fieldnames=self.EVAL_FIELDS,  extrasaction='ignore')
        self._train_w.writeheader()
        self._eval_w.writeheader()

        print(f'Logging to {log_dir}')

    def log_train(self, d: dict):
        row = {k: d.get(k, '') for k in self.TRAIN_FIELDS}
        self._train_w.writerow(row)
        self._train_f.flush()

        phase = d.get('phase', 'tdmpc')
        W = 38
        print('─' * W)
        print(f'  Episode {d["episode"]}   step {d["env_step"]:,}   [{phase}]')
        print('─' * W)
        print(f'  {"Reward":<16}: {d.get("episode_reward", 0):>8.1f}')
        print(f'  {"Horizon":<16}: {d.get("horizon", ""):>8}')
        print(f'  {"Std":<16}: {d.get("std", 0):>8.3f}')
        print(f'  {"Ep time":<16}: {d.get("ep_time", 0):>7.1f}s')
        print(f'  {"Update time":<16}: {d.get("update_time", 0):>7.1f}s')
        if phase == 'o2':
            print(f'  {"Decoder time":<16}: {d.get("decoder_time", 0):>7.1f}s')
            print(f'  {"Decoder loss":<16}: {d.get("decoder_loss", 0):>8.4f}')
        print(f'  {"Total time":<16}: {d.get("total_time", 0):>7.0f}s')
        if d.get('total_loss', '') != '':
            print(f'  {"total_loss":<16}: {d["total_loss"]:>8.3f}')
            print(f'  {"reward_loss":<16}: {d["reward_loss"]:>8.3f}')
            print(f'  {"value_loss":<16}: {d["value_loss"]:>8.3f}')
            print(f'  {"pi_loss":<16}: {d["pi_loss"]:>8.3f}')

    def log_eval(self, d: dict):
        row = {k: d.get(k, '') for k in self.EVAL_FIELDS}
        self._eval_w.writerow(row)
        self._eval_f.flush()

        W = 38
        print('═' * W)
        print(f'  EVAL   episode {d["episode"]}   step {d["env_step"]:,}')
        print(f'  {"Reward":<16}: {d.get("episode_reward", 0):>8.1f}')
        print('═' * W)

    def close(self):
        self._train_f.close()
        self._eval_f.close()


def train(cfg):
    """Main training loop for TD-MPC O2 with DDPG decoder update."""
    assert torch.cuda.is_available(), 'CUDA is required. Use a GPU runtime.'
    set_seed(cfg.seed)

    work_dir = LOG_ROOT / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    logger = CSVLogger(work_dir, cfg)

    env    = make_env(cfg)
    agent  = TDMPC_O2(cfg)
    buffer = ReplayBuffer(cfg)

    print('=' * 60)
    print(OmegaConf.to_yaml(cfg))
    print('=' * 60)
    print(f'Task:                {cfg.task}')
    print(f'Train steps:         {cfg.train_steps * cfg.action_repeat:,}  (env steps)')
    print(f'Obs shape:           {cfg.obs_shape}')
    print(f'Action dim:          {cfg.action_dim}')
    print(f'Latent action dim:   {cfg.latent_action_dim}')
    print(f'Decoder start steps: {cfg.decoder_start_steps * cfg.action_repeat:,}  (env steps)')
    print(f'Seed:                {cfg.seed}')
    print(f'Log dir:             {work_dir}')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time  = time.time()

    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        phase = 'o2' if step >= cfg.decoder_start_steps else 'tdmpc'

        # --- Collect one episode ---
        t_ep = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            if step < cfg.seed_steps:
                action_np = env.action_space.sample()
                action = torch.tensor(action_np, dtype=torch.float32, device=agent.device)
            elif phase == 'tdmpc':
                action = agent.plan(obs, step=step, t0=episode.first)
            else:
                action, *_ = agent.CEM_in_latent(
                    obs, step=step, t0=episode.first, sample_final_action=True
                )
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        assert len(episode) == cfg.episode_length
        buffer += episode
        ep_time = time.time() - t_ep

        # --- Update TOLD model ---
        train_metrics = {}
        update_time = 0.0
        if step >= cfg.seed_steps:
            t_update = time.time()
            train_metrics = update_tdmpc(agent, buffer, step)
            update_time = time.time() - t_update

        # --- Update decoder (DDPG) ---
        decoder_loss = 0.0
        decoder_time = 0.0
        if phase == 'o2':
            t_dec = time.time()
            decoder_loss = update_decoder(agent, buffer, cfg, step)
            decoder_time = time.time() - t_dec

        # --- Log training episode ---
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        logger.log_train({
            'episode':        episode_idx,
            'step':           step,
            'env_step':       env_step,
            'total_time':     time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'horizon':        int(linear_schedule(cfg.horizon_schedule, step)),
            'std':            linear_schedule(cfg.std_schedule, step),
            'ep_time':        ep_time,
            'update_time':    update_time,
            'decoder_time':   decoder_time,
            'decoder_loss':   decoder_loss,
            'phase':          phase,
            **train_metrics,
        })

        # --- Periodic evaluation ---
        if env_step % cfg.eval_freq == 0 and cfg.eval_episodes > 0:
            eval_reward = evaluate(env, agent, cfg.eval_episodes, step)
            logger.log_eval({
                'episode':        episode_idx,
                'env_step':       env_step,
                'episode_reward': eval_reward,
                'total_time':     time.time() - start_time,
            })

        # --- Save model checkpoint ---
        if cfg.get('save_model', False) and env_step % cfg.eval_freq == 0 and env_step > 0:
            ckpt_dir = work_dir / 'models'
            ckpt_dir.mkdir(exist_ok=True)
            agent.save(ckpt_dir / f'model_{env_step}.pt')

    logger.close()
    print('\nTraining complete.')


def make_cfg(task: str, **overrides) -> OmegaConf:
    """
    Build a config programmatically for use in notebooks.

    Example:
        from scripts.train_o2_ddpg import make_cfg
        cfg = make_cfg('walker-walk', seed=1, decoder_start_steps=10000)
        OmegaConf.save(cfg, 'my_cfg.yaml')
        # then: !python scripts/train_o2_ddpg.py cfg=my_cfg.yaml
    """
    old_argv = sys.argv
    sys.argv = ['train_o2_ddpg', f'task={task}'] + [f'{k}={v}' for k, v in overrides.items()]
    try:
        cfg = parse_cfg(CFG_PATH)
        cfg = OmegaConf.merge(OmegaConf.create(O2_DEFAULTS), cfg)
    finally:
        sys.argv = old_argv
    return cfg


def load_cfg() -> OmegaConf:
    """
    Load config with optional custom YAML file.

    Priority (lowest to highest):
      1. tdmpc/cfgs/default.yaml
      2. tdmpc/cfgs/tasks/<domain>.yaml
      3. O2_DEFAULTS
      4. Custom YAML passed as cfg=<path>
      5. Remaining CLI args
    """
    cfg = parse_cfg(CFG_PATH)
    cfg = OmegaConf.merge(OmegaConf.create(O2_DEFAULTS), cfg)
    custom_path = cfg.get('cfg', None)
    if custom_path:
        custom = OmegaConf.load(custom_path)
        cli = OmegaConf.from_cli()
        cli_overrides = OmegaConf.create({k: v for k, v in cli.items() if k != 'cfg'})
        cfg = OmegaConf.merge(cfg, custom, cli_overrides)
    return cfg


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
