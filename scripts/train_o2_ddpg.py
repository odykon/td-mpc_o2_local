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
import time

from omegaconf import OmegaConf
from cfg import parse_cfg
from env import make_env
from algorithm.helper import Episode, ReplayBuffer, linear_schedule
from o2.tdmpc_o2 import TDMPC_O2
from o2.training_utils import set_seed, update_tdmpc, update_decoder
from o2.logger import CSVLogger

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
    # recent-buffer sampling for decoder updates (None = full buffer)
    'dcem_sampling_n':      None,   # None = full buffer; set to int for recency-biased sampling
    'saturation_coeff':     0.0,    # set > 0 to enable saturation penalty
    'use_is_weights':       False,  # apply PER importance-sampling weights to decoder loss
    'dec_grad_clip_norm':   None,   # decoder grad clip threshold (None = no clipping)
}



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


def train(cfg):
    """Main training loop for TD-MPC O2 with DDPG decoder update."""
    assert torch.cuda.is_available(), 'CUDA is required. Use a GPU runtime.'
    set_seed(cfg.seed)

    work_dir = LOG_ROOT / cfg.task / cfg.exp_name / str(cfg.seed)
    logger = CSVLogger(work_dir, cfg)

    use_wandb = cfg.get('use_wandb', False)
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.get('wandb_project', 'td-mpc-o2'),
            entity=cfg.get('wandb_entity', None) or None,
            name=f"{cfg.task}__{cfg.exp_name}__seed{cfg.seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    env    = make_env(cfg)
    agent  = TDMPC_O2(cfg)
    buffer = ReplayBuffer(cfg)

    if cfg.get('load_model', None):
        d = torch.load(cfg.load_model)
        state_dict = d['model'] if 'model' in d else d
        agent.model.load_state_dict(state_dict, strict=False)
        if 'model_target' in d:
            agent.model_target.load_state_dict(d['model_target'], strict=False)
        else:
            agent.model_target.load_state_dict(agent.model.state_dict())
        print(f'Loaded model from {cfg.load_model}')

    if cfg.get('load_buffer', None):
        buffer.__dict__.update(torch.load(cfg.load_buffer, weights_only=False))
        print(f'Loaded buffer from {cfg.load_buffer}')

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
        dec_metrics = {}
        decoder_time = 0.0
        if phase == 'o2':
            t_dec = time.time()
            dec_metrics = update_decoder(agent, buffer, cfg, step)
            decoder_time = time.time() - t_dec
            for iteration, norm in sorted(dec_metrics['grad_tracker']):
                print(f'  DCEM iter {iteration} grad norm: {norm:.6f}')

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
            'phase':          phase,
            **train_metrics,
            **dec_metrics,
        })

        if use_wandb:
            wandb.log({'train/episode_reward': episode.cumulative_reward,
                       'train/phase': 0 if phase == 'tdmpc' else 1,
                       **{f'train/{k}': v for k, v in train_metrics.items()},
                       **{f'decoder/{k}': v for k, v in dec_metrics.items() if k != 'grad_tracker'},
                       }, step=env_step)

        # --- Periodic evaluation ---
        if env_step % cfg.eval_freq == 0 and cfg.eval_episodes > 0:
            eval_reward = evaluate(env, agent, cfg.eval_episodes, step)
            logger.log_eval({
                'episode':        episode_idx,
                'env_step':       env_step,
                'episode_reward': eval_reward,
                'total_time':     time.time() - start_time,
            })
            if use_wandb:
                wandb.log({'eval/episode_reward': eval_reward}, step=env_step)

        # --- Save model checkpoint ---
        if cfg.get('save_model', False) and env_step % cfg.eval_freq == 0 and env_step > 0:
            ckpt_dir = work_dir / 'models'
            ckpt_dir.mkdir(exist_ok=True)
            agent.save(ckpt_dir / f'model_{env_step}.pt')

    agent.save(work_dir / 'final_model.pt')
    logger.close()
    if use_wandb:
        wandb.finish()
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
