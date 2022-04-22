#!/bin/bash
softlearning run_example_local examples.perturbation \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationDQNGraspingRNDPerturbation-v0 \
    --exp-name locobot-navigation-dqn-grasping-rnd-perturbation-test \
    --checkpoint-frequency 10 \
    --gpus 3 \
    --trial-cpus 3 \
    --trial-gpus 0.4 \
    --run-eagerly False \
    --server-port 11112 \

# uses GPU: (gpus + 2) % 4 on newton5 because uhh i really don't know why this sh*t doesn't make any sense
# --gpus 1 --> GPU 3
# --gpus 2 --> GPU 0
# --gpus 3 --> GPU 1
# --gpus 4 --> GPU 2
