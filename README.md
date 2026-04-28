# TD-MPC O2

This repository contains the implementation for my Master's thesis, which extends the [TD-MPC](https://nicklashansen.github.io/td-mpc/) framework (Hansen et al., 2022) with a **learned latent action space** to improve both the sample efficiency and computational efficiency of model-based reinforcement learning in continuous control tasks.

The PyTorch implementation is openly available at: **https://github.com/odykon/td-mpc_o2**

---

## Motivation

Reinforcement learning algorithms can learn to solve complex tasks purely from data, but typically require vast amounts of environment interaction — feasible in simulation, but impractical in most real-world settings. **Model-based RL** addresses this by learning a model of the environment's dynamics and reward function from collected experience, allowing optimization to happen on the model rather than the real environment.

TD-MPC (Hansen et al., 2022) is a state-of-the-art model-based framework that learns a task-oriented latent dynamics model and plans action sequences using the Cross-Entropy Method (CEM), a sampling-based optimizer. While effective, CEM samples candidate action sequences independently — ignoring any structure or correlation between the dimensions of the action sequence. This limits both its statistical efficiency and its computational cost as the planning horizon grows.

---

## Contributions

This thesis proposes advancing TD-MPC by introducing a **learned latent action space** from which CEM can sample, building directly on the Differentiable Cross-Entropy Method (Amos et al., 2018). Instead of sampling raw action sequences, CEM operates over a low-dimensional latent variable `u`, and a learned **action decoder** maps `(u, z) → action sequence`, conditioned on the current latent state `z`.

The key contributions are:

**Latent action space for CEM**
By learning a compact latent space that captures correlations between action sequence dimensions, CEM can explore the space of action sequences more efficiently. This achieves a **16x reduction in the computational cost of the planning procedure** compared to standard TD-MPC, while maintaining or improving performance.

**State-dependent decoder (novel contribution)**
The decoder is conditioned on the environment's latent state `z`, making the latent action space state-dependent. This allows the CEM procedure in latent space to incorporate policy-like behavior — creating a hybrid between explicit planning/optimization and amortized inference of the control objective.

**Two decoder training approaches**
Two strategies for training the decoder are proposed and evaluated:
- *Differentiable CEM (DCEMethod)*: end-to-end training through the planning procedure by replacing hard elite selection with a differentiable soft top-k (via LML projection).
- *Off-policy / on-policy RL*: training the decoder as a policy using DDPG-style gradient ascent on the world model's Q-function, or on-policy policy gradients with a learned value baseline.

Results on simulated continuous control benchmarks (DeepMind Control Suite) under limited-data conditions show **improved sample efficiency over standard TD-MPC** alongside the reduction in planning cost.

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
The decoder is treated as an actor. Off-policy update using transitions from the replay buffer: the decoder maximizes Q(z, decoder(u, z)) with gradients provided by the frozen world model Q-network.

**Policy Gradient with Value Baseline (`PG_withV`)**
On-policy decoder training. Trajectories are collected with CEM_in_latent, storing the CEM distribution parameters and sampled latent actions. The decoder is updated with a REINFORCE-style gradient weighted by advantage estimates `A = R − V(z)`. A separate value network `V(z)` is trained with TD(0) for baseline estimation.

**SAC-style entropy regularization**
Action entropy is included as a training objective alongside the main reward signal, encouraging exploration and preventing premature collapse of the decoder's output.

---

## Repository Structure

```
td-mpc_o2/
├── tdmpc/                    # Original TD-MPC implementation (Hansen et al., 2022)
│   ├── src/
│   │   ├── train.py          # Original training loop
│   │   ├── algorithm/
│   │   │   ├── tdmpc.py      # TOLD model + TDMPC agent
│   │   │   └── helper.py     # Networks, losses, EMA utilities
│   │   ├── env.py            # DMControl wrappers
│   │   └── cfg.py            # OmegaConf config parser
│   └── cfgs/                 # YAML hyperparameter configs
│
├── implementation/           # Thesis contributions (O2 extension)
│   ├── tdmpc_o2.py           # TDMPC_O2 subclass — adds decoder and V-network
│   ├── action_decoder.py     # Decoder and value network construction
│   ├── planning.py           # CEM, CEM_in_latent, DCEMethod
│   ├── training.py           # DDPG and PG decoder update functions
│   ├── pg_training.py        # On-policy PG training functions
│   ├── train_pg.py           # On-policy training loop
│   ├── episode.py            # PGEpisode for on-policy data collection
│   └── logging.py            # Evaluation, metrics, video saving
│
└── lml.py                    # LML soft top-k projection (Amos et al., 2019)
```

---

## Environments

Evaluated on [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) continuous control benchmarks under limited-data conditions:

- **Locomotion**: quadruped-run/walk, walker-run/walk, cheetah-run, hopper-hop, dog-run/walk/trot, humanoid-run/walk/stand, fish-swim
- **Manipulation**: reacher-easy/hard, finger-spin, finger-turn-hard, cup-catch
- **Classic control**: cartpole-swingup, cartpole-swingup-sparse, acrobot-swingup

---

## Acknowledgements

**The Differentiable Cross-Entropy Method** — Brandon Amos, Denis Yarats (2018)
> The foundational paper that introduces learning a latent action space with a decoder and using differentiable CEM for end-to-end training. The central idea this thesis implements and extends within the TD-MPC framework.

**TD-MPC** — Nicklas Hansen, Xiaolong Wang, Hao Su (UC San Diego, 2022)
> *Temporal Difference Learning for Model Predictive Control*
> arXiv: [2203.04955](https://arxiv.org/abs/2203.04955) · [Project page](https://nicklashansen.github.io/td-mpc)

**LML (Limited Multi-Label Projection)** — Amos et al., 2019
> Differentiable soft top-k used inside DCEMethod for gradient-flowing elite selection.
> Implementation from [LocusLab/lml](https://github.com/locuslab/lml).

**DrQv2** — Denis Yarats et al., Facebook Research
> Image augmentation (`RandomShiftsAug`) and DMControl wrappers adapted from [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2).

**TD-MPC2** — Nicklas Hansen et al., 2023
> Successor to the original TD-MPC. [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2)

---

## License

The original TD-MPC code is released under the MIT License. The O2 extensions in this repository are part of a Master's thesis and are openly available for research use.
