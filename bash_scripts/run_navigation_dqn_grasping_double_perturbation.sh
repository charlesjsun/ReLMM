#!/bin/bash
softlearning run_example_local examples.double_perturbation \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationDQNGraspingDoublePerturbation-v0 \
    --exp-name locobot-navigation-dqn-grasping-double-perturbation-test \
    --checkpoint-frequency 10 \
    --gpus 1 \
    --trial-cpus 3 \
    --trial-gpus 0.45 \
    --run-eagerly False \
    --server-port 13336 \
    --temp-dir ~/tmp/ \

# uses GPU: (gpus + 2) % 4 on newton5 because uhh i really don't know why this sh*t doesn't make any sense
# --gpus 1 --> GPU 3
# --gpus 2 --> GPU 0
# --gpus 3 --> GPU 1
# --gpus 4 --> GPU 2
