"""
TD-MPC training script.
Trains a standard TD-MPC agent on a DMControl task using CEM planning
and the standard TOLD model update.

Usage (from repo root):
    python scripts/train_tdmpc.py task=walker-walk seed=1
    python scripts/train_tdmpc.py task=cheetah-run exp_name=myrun eval_freq=10000

Kaggle setup:
    !git clone <your-repo-url>
    %cd <repo-name>
    !pip install dm-control omegaconf
    !python scripts/train_tdmpc.py task=walker-walk

Logs are saved to logs/<task>/<modality>/<exp_name>/<seed>/
  - train.csv : per-episode training metrics
  - eval.csv  : periodic evaluation rewards
  - config.yaml: the full config used for this run
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'tdmpc' / 'src'))

import torch
import numpy as np
import random
import time
import csv

from omegaconf import OmegaConf
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer

torch.backends.cudnn.benchmark = True

CFG_PATH = REPO_ROOT / 'tdmpc' / 'cfgs'
LOG_ROOT = REPO_ROOT / 'logs'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(env, agent, num_episodes: int, step: int) -> float:
    """Run agent in eval mode and return mean episode reward."""
    rewards = []
    for _ in range(num_episodes):
        obs, done, total, t = env.reset(), False, 0.0, 0
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=(t == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            total += reward
            t += 1
        rewards.append(total)
    return float(np.mean(rewards))


class CSVLogger:
    """Logs train and eval metrics to separate CSV files with console output."""

    TRAIN_FIELDS = [
        'episode', 'step', 'env_step', 'total_time', 'episode_reward',
        'ep_time', 'update_time',
        'consistency_loss', 'reward_loss', 'value_loss',
        'pi_loss', 'total_loss', 'weighted_loss', 'grad_norm',
    ]
    EVAL_FIELDS = ['episode', 'env_step', 'episode_reward', 'total_time']

    def __init__(self, log_dir: Path, cfg):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        OmegaConf.save(cfg, log_dir / 'config.yaml')

        self._train_f = open(log_dir / 'train.csv', 'w', newline='')
        self._eval_f = open(log_dir / 'eval.csv', 'w', newline='')
        self._train_w = csv.DictWriter(self._train_f, fieldnames=self.TRAIN_FIELDS, extrasaction='ignore')
        self._eval_w = csv.DictWriter(self._eval_f, fieldnames=self.EVAL_FIELDS, extrasaction='ignore')
        self._train_w.writeheader()
        self._eval_w.writeheader()

        print(f'Logging to {log_dir}')

    def log_train(self, d: dict):
        row = {k: d.get(k, '') for k in self.TRAIN_FIELDS}
        self._train_w.writerow(row)
        self._train_f.flush()
        r = d.get('episode_reward', 0)
        t = d.get('total_time', 0)
        ep_time = d.get('ep_time', 0)
        update_time = d.get('update_time', 0)
        losses = ''
        if d.get('total_loss', '') != '':
            losses = (
                f"  total_loss={d['total_loss']:.3f}"
                f"  reward_loss={d['reward_loss']:.3f}"
                f"  value_loss={d['value_loss']:.3f}"
                f"  pi_loss={d['pi_loss']:.3f}"
            )
        print(f'[train] ep={d["episode"]:5d}  step={d["env_step"]:8,}  reward={r:7.1f}'
              f'  ep={ep_time:.1f}s  update={update_time:.1f}s  total={t:.0f}s{losses}')

    def log_eval(self, d: dict):
        row = {k: d.get(k, '') for k in self.EVAL_FIELDS}
        self._eval_w.writerow(row)
        self._eval_f.flush()
        r = d.get('episode_reward', 0)
        t = d.get('total_time', 0)
        print(f'[eval]  ep={d["episode"]:5d}  step={d["env_step"]:8,}  reward={r:7.1f}  time={t:6.0f}s  <<<')

    def close(self):
        self._train_f.close()
        self._eval_f.close()


def train(cfg):
    """Main training loop for TD-MPC."""
    assert torch.cuda.is_available(), 'CUDA is required. Use a GPU runtime.'
    set_seed(cfg.seed)

    work_dir = LOG_ROOT / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    logger = CSVLogger(work_dir, cfg)

    env = make_env(cfg)
    agent = TDMPC(cfg)
    buffer = ReplayBuffer(cfg)

    print('=' * 60)
    print(OmegaConf.to_yaml(cfg))
    print('=' * 60)
    print(f'Task:        {cfg.task}')
    print(f'Train steps: {cfg.train_steps * cfg.action_repeat:,}  (env steps)')
    print(f'Obs shape:   {cfg.obs_shape}')
    print(f'Action dim:  {cfg.action_dim}')
    print(f'Seed:        {cfg.seed}')
    print(f'Log dir:     {work_dir}')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time = time.time()

    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        # --- Collect one episode ---
        t_ep = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=episode.first)
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
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))
            update_time = time.time() - t_update

        # --- Log training episode ---
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        logger.log_train({
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'ep_time': ep_time,
            'update_time': update_time,
            **train_metrics,
        })

        # --- Periodic evaluation ---
        if env_step % cfg.eval_freq == 0 and cfg.eval_episodes > 0:
            eval_reward = evaluate(env, agent, cfg.eval_episodes, step)
            logger.log_eval({
                'episode': episode_idx,
                'env_step': env_step,
                'episode_reward': eval_reward,
                'total_time': time.time() - start_time,
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
        from scripts.train_tdmpc import make_cfg
        cfg = make_cfg('walker-walk', seed=1, exp_name='my_run')
        cfg.lr = 3e-4
        OmegaConf.save(cfg, 'my_cfg.yaml')
        # then: !python scripts/train_tdmpc.py cfg=my_cfg.yaml
    """
    old_argv = sys.argv
    sys.argv = ['train_tdmpc', f'task={task}'] + [f'{k}={v}' for k, v in overrides.items()]
    try:
        cfg = parse_cfg(CFG_PATH)
    finally:
        sys.argv = old_argv
    return cfg


def load_cfg() -> OmegaConf:
    """
    Load config with optional custom YAML file.

    Priority (lowest to highest):
      1. tdmpc/cfgs/default.yaml
      2. tdmpc/cfgs/tasks/<domain>.yaml
      3. Custom YAML passed as cfg=<path>
      4. Remaining CLI args (e.g. seed=1 exp_name=test)

    Example:
      python scripts/train_tdmpc.py cfg=my_cfg.yaml
      python scripts/train_tdmpc.py cfg=my_cfg.yaml seed=42
    """
    cfg = parse_cfg(CFG_PATH)
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
