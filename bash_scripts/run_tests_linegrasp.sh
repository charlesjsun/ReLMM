#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --policy gaussian \
    --universe gym \
    --domain Tests \
    --task LineGrasping-v0 \
    --exp-name tests-line-grasp-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 11112 \
