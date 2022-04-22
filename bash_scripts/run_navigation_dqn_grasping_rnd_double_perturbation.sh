#!/bin/bash
export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001

softlearning run_example_local examples.double_perturbation \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationDQNGraspingRNDDoublePerturbation-v0 \
    --exp-name locobot-navigation-dqn-grasping-rnd-double-perturbation-test \
    --checkpoint-frequency 10 \
    --gpus 3 \
    --trial-cpus 1 \
    --trial-gpus 3 \
    --run-eagerly False \
    --server-port 11001 \
    --temp-dir ~/tmp/ \
    --env-kwargs '{
        "hi": 2,
        "no": 3,
        "renders": false
    }' \
    --eval-env-kwargs '{
        "hi": 4,
        "uwu": 5
    }' \

# uses GPU: (gpus + 2) % 4 on newton5 because uhh i really don't know why this sh*t doesn't make any sense
# --gpus 1 --> GPU 3
# --gpus 2 --> GPU 0
# --gpus 3 --> GPU 1
# --gpus 4 --> GPU 2
