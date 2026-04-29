env = make_env(cfg)
agent = TDMPC_O2(cfg)
buffer = ReplayBuffer(cfg)

agent.log_alpha_v   = torch.zeros(1, requires_grad=True, device=agent.device)
agent.alpha_v_optim = torch.optim.Adam([agent.log_alpha_v], lr=cfg.lr)
agent.target_entropy = 0.2*cfg.action_dim   # positive target, tune as needed

episode_idx, start_time = 0, time.time()
print("\n" + "=" * 50)
print("\033[1m🚀 Training Configuration\033[0m")
print("=" * 50)
for key, value in agent.cfg.items():
    print(f"{key:25}: {value}")
print("=" * 50 + "\n")
save_dir = l.make_save_dir_path(agent.cfg, base_dir = RESULTS_DIR)
print("Saving results to:", save_dir)

all_metrics = []

for step in range(0, cfg.train_steps, cfg.episode_length):

    obs     = env.reset()
    episode = PGEpisode(cfg, obs)       # ← PGEpisode, not Episode
    current_step     = 0
    half_time_reward = 0.0
    episode_start_time = time.time()
        # ---- Rollout -------------------------------------------------------
    while not episode.done:
        if step < cfg.seed_steps:
            action_np = env.action_space.sample()
            action    = torch.tensor(action_np, dtype=torch.float32,
                                      device=agent.device)
        else:
            action, u_mean, u_std, latent_action, log_prob = (
                agent.CEM_in_latent(
                    obs, step=step, t0=episode.first,
                    seed=None, sample_final_action=True,
                )
            )
            # Store on-policy data for this step
            episode.add_pg(log_prob, u_mean, u_std, latent_action)

        obs, reward, done, _ = env.step(action.cpu().numpy())
        current_step        += 1
        if current_step <= 500:
            half_time_reward += reward

        episode += (obs, action, reward, done)
    episode.finalize()
    episode_end_time = time.time()


    buffer+=episode

    horizon = int(linear_schedule(cfg.horizon_schedule,step))
    episode_metrics = {
        'Episode_no:': int(step/cfg.episode_length),
        'Reward': episode.cumulative_reward,
        'Horizon': int(linear_schedule(cfg.horizon_schedule,step)),
        'Std:': linear_schedule(cfg.std_schedule,step),
        'Duration:': episode_end_time - episode_start_time,
    }

    print("Buffer idx:", buffer.idx,", buffer is full:", buffer._full)

    print("\n  Episode Summary")
    print("-" * 25)
    for k, v in episode_metrics.items():
        print(f"  {k:15}: {v}")


    train_metrics = {}
    decoder_metrics = {}


    #     On-policy O2 update
    if step >= cfg.seed_steps:
        t0 = time.time()
        decoder_metrics = update_decoder_pg(agent, episode, step)
        train_metrics["decoder_update_s"] = time.time() - t0

        print("  Decoder Update Metrics:")
        for k, v in decoder_metrics.items():
            print(f"    {k:20}: {v:.4f}")


    #     TD-MPC standard update
    if step >= cfg.seed_steps:
        model_update_start_time = time.time()
        train_metrics = update_tdmpc(agent, buffer, step)
        model_update_end_time = time.time()

    """
    if (step>= cfg.seed_steps):
        decoder_update_start_time = time.time()
        decoder_loss = update_decoder(agent, buffer, cfg, step)
        train_metrics['Decoder_loss'] = decoder_loss
        decoder_update_end_time = time.time()
    """

    print("  Training Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k:15}: {v}")
