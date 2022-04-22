#!/bin/bash
export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001

softlearning run_example_local examples.dual_perturbation \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationDQNGraspingDualPerturbation-v0 \
    --exp-name nav-dqn-reset \
    --checkpoint-frequency 10 \
    --gpus 6 \
    --trial-cpus 1 \
    --trial-gpus 0.45 \
    --max-failures 0 \
    --run-eagerly False \
    --server-port 11054 \
    --temp-dir ~/tmp/ \
    --env-kwargs '{
        "grasp_perturbation": "rnd",
        "nav_perturbation": "rnd",
        "trajectory_log_path": "./trajectory/",
        "renders": false,
        "step_duration": 0.0,
        "force_reset": false,
        "force_reset_ep_len": 250,
        "grasp_algorithm": "dqn",
        "do_grasp_eval": true,
        "use_shared_data": false,
    }' \
    --eval-env-kwargs '{
        "renders": false,
        "step_duration": 0.0
    }' \

