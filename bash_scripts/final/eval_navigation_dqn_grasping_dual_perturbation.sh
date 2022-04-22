#!/bin/bash
python -m examples.dual_perturbation.simulate_policy \
    ./nohup_output/results/rnd_rnd_1/checkpoint_101/ \
    --max-path-length 2000 \
    --num-rollouts 1 \
    --env-kwargs '{}' \
