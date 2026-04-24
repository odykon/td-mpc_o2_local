# TD-MPC O2

This repository contains the implementation for my Master's thesis at the National Technical University of Athens, which extends the [TD-MPC](https://nicklashansen.github.io/td-mpc/) framework (Hansen et al., 2022) with a learned latent action space for model-based reinforcement learning in continuous control tasks.

Thesis: [Latent Space Planning with Model-Based Reinforcement Learning](http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/20101)

PyTorch implementation: [github.com/odykon/td-mpc_o2](https://github.com/odykon/td-mpc_o2)

---

## Motivation

Model-based RL improves sample efficiency by learning a model of the environment's dynamics and reward function, then using it for planning — evaluating candidate actions under the model rather than executing them in the real environment.

One such method is the Temporal Difference Model Predictive Control (TD-MPC) framework (Hansen et al., 2022) which employs a learned model to perform planning with the Cross-Entropy Method (CEM), a sampling-based optimizer. Although TD-MPC achieves state-of-the-art performance in continuous control tasks with improved data efficiency, we propose advancing this approach further by introducing a learned latent action space from which the CEM can sample. This builds directly on the Differentiable Cross-Entropy Method (Amos et al., 2018), which proposes learning a latent action space through a parameterized decoder.

Through the latent action space we can leverage correlations between the action sequences that the CEM otherwise ignores and significantly improve both the performance and the computational efficiency of the method. Furthermore, we propose conditioning this decoder on the environment state, thus making the latent action space state-dependent. This allows the CEM procedure in latent space to incorporate policy-like behavior, creating a hybrid between explicit optimization and amortized inference of the control objective. We introduce two training approaches for this latent space and evaluate the resulting algorithm on simulated continuous control benchmarks under limited-data conditions, achieving a 16x reduction in the computational cost of the planning procedure while demonstrating improved sample efficiency over standard TD-MPC.

---

## Contributions

**State-dependent latent action space.** A decoder network is introduced that maps a latent action vector `u` together with the current latent state `z` to a full action sequence:

```
f_dec: (u, z) → action sequence   [horizon × action_dim]
```

CEM searches over `u` rather than directly over action sequences. The state-conditioning is a key design choice: by conditioning on `z`, the latent space becomes state-dependent, allowing policy information to be embedded directly into the CEM sampling procedure. This bridges the advantages of amortizing the optimisation problem — having a global policy — with those of directly solving it via CEM, and goes beyond the approach in the original DCEM paper which used a state-independent decoder.

**Open-source DCEM decoder training.** The DCEM paper (Amos and Yarats, 2020) proposed training a latent action space through the differentiable CEM procedure but did not release an implementation of this training method. This repository provides the first open-source implementation of this approach, ported from the PlaNet architecture used in the original paper to TD-MPC.

**On-policy decoder training with a derived policy gradient.** A second training strategy is introduced that trains the decoder directly from environment reward signals rather than from the world model alone. The thesis derives a tractable policy gradient expression for the full DCEM procedure treated as a stochastic parameterised policy in the latent action space. The gradient decomposes into a likelihood-ratio term over the latent action distribution and a reward gradient term through the decoder mapping. Entropy regularisation over decoded action sequences is introduced to prevent latent action space collapse during training, formulated as a constrained optimisation problem with an auto-tuned temperature following the SAC framework.

The result is a 16x reduction in CEM samples (from 512 to 32) alongside improved sample efficiency on the evaluated tasks.

---

## Method

### Planning

The CEM objective over raw action sequences is replaced by a search over latent actions. For a given encoded state `z`, the optimisation problem becomes:

```
û = argmax_{u ∈ U} G_φ(f_dec(u, z), z)
```

where `G_φ` is the world model's multi-step return estimate. CEM is run in `U` rather than in the raw action space.

**CEM in Latent Space (`CEM_in_latent`)** implements this with hard top-k elite selection. It is used for environment interaction during training.

**Differentiable CEM (`DCEMethod`)** replaces the hard top-k with the LML soft top-k projection (Amos et al., 2019), making the elite selection differentiable. This allows gradients to flow back through the entire planning procedure into the decoder during training. All sampling steps use the reparameterisation trick.

### Decoder Architecture and Initialisation

The decoder is a two-layer MLP with a Tanh output:

```
[u, z] → Linear → ReLU → Linear → Tanh → [horizon × action_dim]
```

The decoder is initialised so that each latent action dimension maps approximately to one action dimension per horizon step, using the identity `ReLU(x) - ReLU(-x) = x` to implement a near-identity through the ReLU bottleneck. This ensures consistency between the DCEM policy and the world model's Q-function at the start of training.

### Off-Policy Decoder Training

The decoder is trained off-policy by maximising the world model's return estimate `G_φ` through the full DCEM procedure. Latent states are sampled from the replay buffer, DCEM is unrolled to obtain the optimised distribution mean, the mean is decoded into an action sequence, and gradients are backpropagated through all sampling steps of DCEM. The world model is held frozen during decoder updates.

This corresponds to a DDPG-style formulation where the full DCEM procedure acts as the actor and `G_φ` acts as the critic. Because the approach does not depend on which actions were taken or what rewards were received during data collection, it is entirely off-policy and can use any previously collected experience.

The key advantage is stability — the world model provides a low-variance training signal that guides the decoder without the noise inherent in direct environment interaction. The key limitation is that the decoder is only as good as the world model, and can exploit model errors if overtrained.

### On-Policy Decoder Training

To reduce dependence on the world model and train the decoder directly from environment rewards, a policy gradient approach is derived. The full DCEM procedure is treated as a stochastic parameterised policy in the latent action space `q_θ(u|z)`, and the gradient of the expected return with respect to the decoder parameters `θ` is derived. The key insight is that because the decoder parameters appear in both the latent policy distribution and in the mapping from latent to ground-truth actions, the gradient has two terms:

```
∇_θ J = E[∇_θ log q_θ(u|z) · Q(z, a)] + E[γ^t · ∇_θ r(s, a)]
```

where `a = f_dec(u, z)`. The first term is a likelihood-ratio gradient over the latent distribution; the second is a reward gradient through the decoder mapping, which is non-zero because the decoder parameters affect which ground-truth action is applied to the environment and therefore what reward is received. The reward gradient term is included in the implementation and weighted by a tunable coefficient; in the presented experiments it is set to zero.

In practice, a one-step advantage estimate replaces the full return:

```
A(z, a) ≈ r + V^φ(z') - V^φ(z)
```

where `V^φ` is a separate value network trained with TD(0). Gradients are backpropagated through all sampling steps of DCEM. The TOLD model is updated after decoder updates are complete in each episode, to limit deviation between the model used during data collection and the model used during training.

**Entropy regularisation.** Training the decoder from environment rewards can cause latent action space collapse — the decoder prematurely concentrates on a small region of the action space, trapping CEM in suboptimal regions. To prevent this, a regularisation term maximises the average differential entropy of decoded action sequences. At each update, `N` latent actions are sampled from the current search distribution, decoded into action sequences, and a multivariate Gaussian is fitted to the decoded actions at each horizon step. The average differential entropy across timesteps is maximised as a batch-level objective, allowing the latent space to explore more in states where the optimal action is uncertain and less where good actions are already known. The entropy temperature coefficient is auto-tuned via a SAC-style Lagrangian constraint targeting a minimum entropy level.

PPO ideas are also explored (`action_decoder_PPO`), adding a KL penalty between the current and behaviour policy distributions to constrain how far the decoder moves per update.

---

## Experimental Results

Evaluated on [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) under limited-data conditions (20,000–40,000 environment steps):

Tasks: cheetah-run, cartpole-swingup

The latent action space achieves improved sample efficiency over standard TD-MPC while reducing the number of CEM samples from 512 to 32. Full experimental details and learning curves are available in the thesis.

---

## Repository Structure

```
td-mpc_o2/
├── tdmpc/                     # Original TD-MPC (Hansen et al., 2022), unmodified
│   ├── src/
│   │   ├── algorithm/
│   │   │   ├── tdmpc.py       # TOLD model and TDMPC agent
│   │   │   └── helper.py      # Networks, losses, EMA utilities
│   │   ├── env.py             # DMControl environment wrappers
│   │   └── cfg.py             # OmegaConf config parser
│   └── cfgs/                  # YAML hyperparameter configs
│
├── implementation/            # This contribution
│   ├── tdmpc_o2.py            # TDMPC_O2 subclass: adds decoder and V-network
│   ├── action_decoder.py      # Decoder architecture and initialisation
│   ├── planning.py            # CEM_in_latent and DCEMethod
│   ├── training.py            # Off-policy DDPG-style decoder update
│   ├── pg_training.py         # On-policy PG decoder update and entropy regularisation
│   ├── ppo_training.py        # PPO-style decoder update
│   ├── episode.py             # Episode class for on-policy data collection
│   ├── train_pg.py            # On-policy training loop
│   └── logging.py             # Evaluation, metrics, video saving
│
└── lml.py                     # LML soft top-k projection (Amos et al., 2019)
```

The original TD-MPC code is entirely unmodified. `TDMPC_O2` subclasses `TDMPC` and attaches the decoder and new methods at initialisation, requiring no changes to the base class.


## References

**The Differentiable Cross-Entropy Method**
Brandon Amos, Denis Yarats. ICML 2020.
The paper introducing differentiable CEM and the idea of training a latent action space through the planning procedure. This work implements that training method (not previously open-sourced) and extends it with a state-dependent decoder and on-policy training within the TD-MPC framework.
[arxiv.org/abs/1909.12830](https://arxiv.org/abs/1909.12830)

**TD-MPC: Temporal Difference Learning for Model Predictive Control**
Nicklas Hansen, Xiaolong Wang, Hao Su. ICML 2022.
[arxiv.org/abs/2203.04955](https://arxiv.org/abs/2203.04955)

**The Limited Multi-Label Projection Layer**
Brandon Amos, Vladlen Koltun, J. Zico Kolter. 2019.
Differentiable soft top-k projection used inside DCEMethod.
[github.com/locuslab/lml](https://github.com/locuslab/lml)

**Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning**
Tuomas Haarnoja et al. ICML 2018.
Maximum entropy RL framework used for entropy regularisation and temperature auto-tuning.
[arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)

**Proximal Policy Optimization Algorithms**
John Schulman et al. 2017.
[arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

**Policy Gradient Methods for Reinforcement Learning with Function Approximation**
Richard S. Sutton et al. NeurIPS 1999.
Policy gradient theorem used in the on-policy gradient derivation.

**DrQv2**
Denis Yarats et al., Meta AI Research.
Image augmentation and DMControl wrappers adapted from [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2).

---

## License

The original TD-MPC code is released under the MIT License. The contributions in `implementation/` are part of a Master's thesis at NTUA and are openly available for research use.
