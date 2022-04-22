#!/bin/bash
export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001

export CUDA_VISIBLE_DEVICES=0

softlearning run_example_local examples.dual_perturbation \
    --algorithm SAC \
    --policy gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationDQNGraspingDualPerturbationOracle-v0 \
    --exp-name nav-auto-vac-oracle-reset-respawn \
    --checkpoint-frequency 10 \
    --trial-cpus 1 \
    --trial-gpus 0.0 \
    --max-failures 0 \
    --run-eagerly False \
    --server-port 11051 \
    --temp-dir ~/tmp/ \
    --env-kwargs '{
        "grasp_perturbation": "none",
        "nav_perturbation": "none",
        "grasp_algorithm": "vacuum",
        "use_auto_grasp": true,
        "do_respawn": true,
        "respawn_radius": 1.5,
        "do_grasp_eval": false,
        "use_shared_data": false,
        "force_reset": true,
        "force_reset_ep_len": 500,
        "trajectory_log_path": "./trajectory/",
        "renders": true,
        "step_duration": 0.0
    }' \
    --eval-env-kwargs '{
        "do_respawn": false,
        "renders": false,
        "step_duration": 0.0
    }' \

