#!/bin/bash
# Run O2-DDPG experiments across 5 environments x 5 seeds.
#
# Step counts are in raw MuJoCo steps (env interactions before action_repeat).
# train_o2_ddpg.py divides each by the task's action_repeat automatically.
#
# Usage:
#   bash scripts/run_experiments.sh
#   bash scripts/run_experiments.sh 2>&1 | tee logs/run_experiments.log

# --- Experiment parameters (MuJoCo steps) ---
MUJOCO_TRAIN_STEPS=500000
MUJOCO_DECODER_START=150000
MUJOCO_LATENT_START=200000

TASKS=(cartpole-swingup cheetah-run walker-walk hopper-hop finger-spin)
SEEDS=(1 2 3 4 5)

SCRIPT="$(dirname "$0")/train_o2_ddpg.py"
TOTAL=$(( ${#TASKS[@]} * ${#SEEDS[@]} ))
RUN=0

for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN=$(( RUN + 1 ))
        echo "========================================"
        echo "[$RUN/$TOTAL] task=$task seed=$seed"
        echo "========================================"
        python "$SCRIPT" \
            task="$task" \
            seed="$seed" \
            mujoco_train_steps="$MUJOCO_TRAIN_STEPS" \
            mujoco_decoder_start_steps="$MUJOCO_DECODER_START" \
            mujoco_latent_start_steps="$MUJOCO_LATENT_START"
        if [ $? -ne 0 ]; then
            echo "ERROR: $task seed=$seed failed. Continuing..."
        fi
    done
done

echo "All $TOTAL runs complete."
