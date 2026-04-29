# Repo: TD-MPC O2

Thesis project extending TD-MPC with a latent action space decoder (the "O2" extension).
The goal is to train agents on DMControl tasks using CEM planning in a learned latent action space.

## Repo Structure

```
scripts/          # Runnable training entry points
  train_tdmpc.py  # Standard TD-MPC training (base agent, CEM planning)

o2/               # O2 extension — latent action space decoder
  action_decoder.py   # Decoder MLP architecture and initialisation
  planning.py         # DCEMethod (differentiable CEM) and CEM_in_latent
  decoder_updates.py  # All decoder update strategies (DDPG, PG, PPO)
  training_utils.py   # Shared loop utilities: collect_episode, update_tdmpc, update_decoder, update_decoder_pg
  episode.py          # PGEpisode — extends Episode with on-policy fields
  eval_utils.py       # Evaluation, metrics, video saving

tdmpc/            # Original TD-MPC implementation (Hansen et al., 2022) — do not modify
  src/            # Core source: TDMPC agent, TOLD model, ReplayBuffer, env wrappers, cfg parsing
  cfgs/           # YAML configs: default.yaml, tasks/<domain>.yaml

lml.py            # LML soft top-k projection (Amos et al.) — external, do not modify

logs/             # Training outputs (gitignored)
results/          # Evaluation outputs (gitignored)
```

## Key Design Concepts

**Standard TD-MPC** (`scripts/train_tdmpc.py`):
- Uses `TDMPC` agent from `tdmpc/src/algorithm/tdmpc.py`
- Plans with standard CEM via `agent.plan()`
- Updates world model via `agent.update()` (wrapped in `update_tdmpc` from `training_utils`)
- Logs to `logs/<task>/<modality>/<exp_name>/<seed>/train.csv` and `eval.csv`

**O2 Extension** (`o2/`):
- Adds a learned action decoder that maps latent actions `u` → action sequences
- Planning happens in latent action space via `DCEMethod` (differentiable, for training) or `CEM_in_latent` (hard top-k, for rollouts)
- Decoder can be trained with DDPG, PG, or PPO strategies — all in `decoder_updates.py`
- `training_utils.py` provides reusable loop functions shared across all O2 training scripts

## Config System

Configs are loaded by `parse_cfg` from `tdmpc/src/cfg.py`. Priority order (lowest to highest):
1. `tdmpc/cfgs/default.yaml` — base defaults
2. `tdmpc/cfgs/tasks/<domain>.yaml` — task-specific (mainly `action_repeat`)
3. Custom YAML passed as `cfg=my_cfg.yaml`
4. CLI args e.g. `task=walker-walk seed=1`

Use `make_cfg` from `scripts/train_tdmpc.py` to build a config programmatically in notebooks:
```python
from scripts.train_tdmpc import make_cfg
from omegaconf import OmegaConf
cfg = make_cfg('walker-walk', seed=1)
OmegaConf.save(cfg, 'my_cfg.yaml')
# then: !python scripts/train_tdmpc.py cfg=my_cfg.yaml
```

Disable evaluations by setting `eval_episodes=0` in the config.

## Kaggle Usage

```bash
!git clone <repo-url>
%cd <repo-name>
!pip install dm-control omegaconf
!python scripts/train_tdmpc.py task=walker-walk seed=1
```

GPU runtime must be enabled. `MUJOCO_GL=egl` is set automatically by the script.

## Conventions

- Training scripts live in `scripts/` — one file per training procedure
- Shared utilities (episode collection, update loops) live in `o2/training_utils.py`
- Decoder update functions are methods bound to `TDMPC_O2` — they take `self` as first arg
- `tdmpc/` is upstream code — never modify it
- `lml.py` is external (Amos et al.) — never modify it
- `logs/` and `results/` are gitignored — never commit training outputs

## Planned Scripts

Future training scripts to be added to `scripts/`:
- `train_o2_ddpg.py` — decoder trained with DDPG update
- `train_o2_pg.py`   — decoder trained with on-policy PG
- `train_o2_ppo.py`  — decoder trained with PPO-KL
All will import shared utilities from `o2/training_utils.py`.
