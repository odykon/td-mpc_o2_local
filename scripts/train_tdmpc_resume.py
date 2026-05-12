"""
train_tdmpc_resume.py — TDMPC baseline continuation from intermediate checkpoint.

Loads the intermediate model and buffer saved by train_o2_phased.py at
mujoco_step_offset (default: 20k MuJoCo), then continues standard TDMPC
training for mujoco_resume_steps (default: 20k MuJoCo) more interactions.

Logs to the same W&B project with group=task and exp_name='tdmpc_baseline',
so reward curves start at env_step=20k and overlay directly with the O2 runs.

Usage:
    python scripts/train_tdmpc_resume.py task=walker-walk seed=1
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import re
import sys
import glob
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tdmpc' / 'src'))

import torch
import time
import wandb

from omegaconf import OmegaConf
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer, linear_schedule
from o2.training_utils import set_seed, update_tdmpc

torch.backends.cudnn.benchmark = True

CFG_PATH = REPO_ROOT / 'tdmpc' / 'cfgs'

RESUME_DEFAULTS = {
    'mujoco_step_offset':  20000,  # must match mujoco_latent_start_steps in phased config
    'mujoco_resume_steps': 20000,
    'told_updates':        500,
    'seed_steps':          0,      # model is already trained, skip random exploration
    'exp_name':            'tdmpc_baseline',
    'wandb_project':       'TDMPC_O2',
    'wandb_entity':        'odysseaskon-national-technical-university-of-athens',
}


def train(cfg):
    assert torch.cuda.is_available(), 'CUDA is required.'
    set_seed(cfg.seed)

    task_safe = cfg.task.replace('-', '_')

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.task,
        name=f"{cfg.exp_name}__seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    env    = make_env(cfg)
    agent  = TDMPC(cfg)
    buffer = ReplayBuffer(cfg)

    # ── Download and load intermediate model + buffer from W&B ───────────────
    api = wandb.Api()

    model_tmp = tempfile.mkdtemp()
    try:
        art = api.artifact(f"{cfg.wandb_entity}/{cfg.wandb_project}/intermediate_{task_safe}_seed{cfg.seed}:latest")
        art.download(root=model_tmp)
        model_file = glob.glob(os.path.join(model_tmp, '*.pt'))[0]
        ckpt = torch.load(model_file, map_location=agent.device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        agent.model.load_state_dict(state_dict, strict=False)
        if 'model_target' in ckpt:
            agent.model_target.load_state_dict(ckpt['model_target'], strict=False)
        else:
            agent.model_target.load_state_dict(agent.model.state_dict())
        print(f'Loaded intermediate model (seed {cfg.seed}).')
    finally:
        shutil.rmtree(model_tmp, ignore_errors=True)

    buffer_tmp = tempfile.mkdtemp()
    try:
        art = api.artifact(f"{cfg.wandb_entity}/{cfg.wandb_project}/intermediate_buffer_{task_safe}_seed{cfg.seed}:latest")
        art.download(root=buffer_tmp)
        buffer_file = glob.glob(os.path.join(buffer_tmp, '*.pth'))[0]
        buffer.__dict__.update(torch.load(buffer_file, weights_only=False))
        print(f'Loaded intermediate buffer (seed {cfg.seed}).')
    finally:
        shutil.rmtree(buffer_tmp, ignore_errors=True)

    print('=' * 60)
    print(f'Task:                {cfg.task}')
    print(f'Resuming from:       {cfg.mujoco_step_offset:,}  MuJoCo')
    print(f'Running for:         {cfg.mujoco_resume_steps:,}  MuJoCo')
    print(f'TOLD updates/ep:     {cfg.told_updates}')
    print(f'Seed:                {cfg.seed}')
    print('=' * 60 + '\n')

    episode_idx = 0
    start_time  = time.time()

    for step in range(cfg.step_offset,
                      cfg.step_offset + cfg.resume_steps + cfg.episode_length,
                      cfg.episode_length):

        t_ep = time.time()
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        buffer += episode
        episode_idx += 1
        ep_time = time.time() - t_ep

        t_update = time.time()
        train_metrics = update_tdmpc(agent, buffer, step)
        update_time = time.time() - t_update

        env_step   = int(step * cfg.action_repeat)
        horizon    = int(linear_schedule(cfg.horizon_schedule, step))
        std        = linear_schedule(cfg.std_schedule, step)
        total_time = time.time() - start_time

        SEP = '─' * 42
        print(f'\n{SEP}')
        print(f'  Episode {episode_idx}   step {env_step:,}   [tdmpc_baseline]')
        print(SEP)
        def row(label, val):
            print(f'  {label:<22}: {val}')
        row('Reward',      f'{episode.cumulative_reward:>10.1f}')
        row('Horizon',     f'{horizon:>10d}')
        row('Std',         f'{std:>10.3f}')
        row('Ep time',     f'{ep_time:>9.1f}s')
        row('Update time', f'{update_time:>9.1f}s')
        row('Total time',  f'{total_time:>9.0f}s')
        for k, v in train_metrics.items():
            row(k, f'{v:>10.4f}')

        wandb.log({
            'episode':              episode_idx,
            'train/episode_reward': episode.cumulative_reward,
            'train/horizon':        horizon,
            'train/std':            std,
            **{f'train/{k}': v for k, v in train_metrics.items()},
        }, step=env_step)

    wandb.finish()
    print(f'\nDone. Total time: {(time.time() - start_time) / 60:.1f} min')


def load_cfg() -> OmegaConf:
    cfg = parse_cfg(CFG_PATH)
    cfg = OmegaConf.merge(OmegaConf.create(RESUME_DEFAULTS), cfg)

    custom_path = cfg.get('cfg', None)
    if custom_path:
        custom = OmegaConf.load(custom_path)
        cli    = OmegaConf.from_cli()
        cli_overrides = OmegaConf.create({k: v for k, v in cli.items() if k != 'cfg'})
        cfg = OmegaConf.merge(cfg, custom, cli_overrides)

    for k, v in cfg.items():
        if isinstance(v, str):
            m = re.match(r'^(\d+)([+\-*/])(\d+)$', v)
            if m:
                result = eval(m.group(1) + m.group(2) + m.group(3))
                cfg[k] = int(result) if isinstance(result, float) and result.is_integer() else result

    ar = cfg.action_repeat
    cfg.step_offset  = int(cfg.mujoco_step_offset)  // ar
    cfg.resume_steps = int(cfg.mujoco_resume_steps) // ar
    cfg.train_steps  = cfg.step_offset + cfg.resume_steps  # for buffer capacity + schedules

    return cfg


if __name__ == '__main__':
    cfg = load_cfg()
    train(cfg)
