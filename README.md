# TD-MPC O2

This repository contains the implementation for my Master's thesis, which extends the [TD-MPC](https://nicklashansen.github.io/td-mpc/) framework (Hansen et al., 2022) with a **learned latent action space** to improve both the sample efficiency and computational efficiency of model-based reinforcement learning in continuous control tasks.

---
## Contributions

This thesis proposes advancing TD-MPC by introducing a **learned latent action space** from which CEM can sample, building directly on the Differentiable Cross-Entropy Method (Amos et al., 2018). Instead of sampling raw action sequences, CEM operates over a low-dimensional latent variable `u`, and a learned **action decoder** maps `(u, z) → action sequence`, conditioned on the current latent state `z`.

The key contributions are:

**Latent action space for CEM**
By learning a latent space that captures correlations between action sequence dimensions, CEM can explore the space of action sequences more efficiently. We show that this can achieve a **16x reduction in the computational cost of the planning procedure** compared to standard TD-MPC, while maintaining or improving performance.

**State-dependent decoder**
The decoder is conditioned on the environment's latent state `z`, making the latent action space state-dependent. This allows the CEM procedure in latent space to incorporate policy-like behavior — creating a hybrid between explicit planning/optimization and amortized inference of the control objective.

**Two decoder training approaches**
Using the Differentiable Cross-Entropy Method (DCEM), the decoder is trained end-to-end through the complete planning procedure via two distinct approaches
Two strategies for training the decoder are proposed and evaluated:
- *Model-Based, Off-policy RL*: trained to maximise a model estimate
- *REINFORCE-style, On-policy RL*: trained to directly maximise rewards

Results on simulated continuous control benchmarks (DeepMind Control Suitem, Cheetah-run and Cartpole-swingup) under limited-data conditions show **improved sample efficiency over standard TD-MPC** alongside the reduction in planning cost.

---

## Algorithms

### Planning & Control

**Cross-Entropy Method (CEM)**
Samples candidate action sequences from a Gaussian, evaluates them under the world model, selects the top-k elites, and refits the Gaussian — iterated for a fixed number of rounds. Used in the original TD-MPC baseline.

**CEM in Latent Space (`CEM_in_latent`)**
CEM operating over the low-dimensional latent action `u` rather than raw action sequences. The decoder expands each sampled `u` into a full action sequence conditioned on the current state. Reduces the search dimensionality significantly.

**Differentiable CEM (`DCEMethod`)**
Based on Amos et al. (2018). Replaces the hard top-k elite selection with a differentiable soft top-k via the Limited Multi-Label (LML) projection, so gradients flow back through the planning process into the decoder. This enables end-to-end training of the decoder directly through the planner.

**Model Predictive Control (MPC)**
The agent replans at every timestep using the learned world model. Only the first action of the planned sequence is executed; the rest is discarded and replanning occurs on the next step.

### Reinforcement Learning

**Temporal Difference (TD) Learning**
Core value learning throughout. The Q-function is trained with multi-step TD targets and a slow-moving EMA target network for stability. The world model (encoder, dynamics, reward) is jointly trained.

**DDPG-Style Decoder Update (`action_decoder_DDPG_update`)**
The planning procedure with the decoder is treated as an actor. Off-policy update using transitions from the replay buffer: the decoder maximizes Q(z, decoder(u, z)) with gradients provided by the frozen world model Q-network.

**Policy Gradient with Value Baseline (`PG_withV`)**
On-policy decoder training. Trajectories are collected with CEM_in_latent, storing the CEM distribution parameters and sampled latent actions. The decoder is updated with a REINFORCE-style gradient weighted by advantage estimates `A = R − V(z)`. A separate value network `V(z)` is trained with TD(0) for baseline estimation.

**SAC-style entropy regularization**
Action entropy is included as a training objective alongside the main reward signal, encouraging exploration and preventing premature collapse of the decoder's output.

---

## Repository Structure

```
td-mpc_o2/
├── scripts/                  # Runnable training entry points
│   ├── train_tdmpc.py        # Standard TD-MPC training (base agent, CEM planning)
│   └── train_o2_ddpg.py      # O2 training with DDPG decoder update (two-phase loop)
│
├── o2/                       # Thesis contributions (O2 extension)
│   ├── tdmpc_o2.py           # TDMPC_O2 subclass: adds decoder, value network, and O2 planners
│   ├── action_decoder.py     # Decoder and value network construction
│   ├── planning.py           # DCEMethod (differentiable CEM) and CEM_in_latent
│   ├── decoder_updates.py    # All decoder update strategies (DDPG, PG, PPO)
│   ├── training_utils.py     # Shared loop utilities for training scripts
│   ├── episode.py            # PGEpisode for on-policy data collection
│   ├── eval_utils.py         # Evaluation, metrics, video saving
│   └── logger.py             # CSV logger for train/eval metrics (shared by all scripts)
│
├── tdmpc/                    # Original TD-MPC implementation (Hansen et al., 2022)
│   ├── src/
│   │   ├── algorithm/
│   │   │   ├── tdmpc.py      # TOLD model + TDMPC agent
│   │   │   └── helper.py     # Networks, losses, EMA utilities
│   │   ├── train.py          # Original TD-MPC training entry point
│   │   ├── logger.py         # Original TD-MPC logger
│   │   ├── env.py            # DMControl wrappers
│   │   └── cfg.py            # OmegaConf config parser
│   └── cfgs/                 # YAML hyperparameter configs
│
└── lml.py                    # LML soft top-k projection (Amos et al., 2019)
```


## Acknowledgements

**The Differentiable Cross-Entropy Method** — Brandon Amos, Denis Yarats (2018)
> The foundational paper that introduces learning a latent action space with a decoder and using differentiable CEM for end-to-end training. The central idea this thesis implements and extends within the TD-MPC framework.

**TD-MPC** — Nicklas Hansen, Xiaolong Wang, Hao Su (UC San Diego, 2022)
> *Temporal Difference Learning for Model Predictive Control*
> arXiv: [2203.04955](https://arxiv.org/abs/2203.04955) · [Project page](https://nicklashansen.github.io/td-mpc)

**LML (Limited Multi-Label Projection)** — Amos et al., 2019
> Differentiable soft top-k used inside DCEMethod for gradient-flowing elite selection.
> Implementation from [LocusLab/lml](https://github.com/locuslab/lml).

---

## License

The original TD-MPC code is released under the MIT License. The O2 extensions in this repository are part of a Master's thesis and are openly available for research use.
