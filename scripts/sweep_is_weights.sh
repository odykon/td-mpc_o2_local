#!/bin/bash
# Sweep: IS weights on vs off, 2 seeds each (sequential, single GPU)
# Usage: bash scripts/sweep_is_weights.sh

BASE="python3 scripts/train_o2_ddpg.py cfg=cfgs/train_o2_ddpg.yaml"

for use_is_weights in true false; do
    for seed in 1 2; do
        echo "=== use_is_weights=${use_is_weights}  seed=${seed} ==="
        $BASE use_is_weights=$use_is_weights seed=$seed exp_name="is_weights_${use_is_weights}"
    done
done
