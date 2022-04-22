#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task MixedNavigationReach-v0 \
    --exp-name locobot-mixed-navigation-reach-test \
    --checkpoint-frequency 10 \
    --trial-cpus 6 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 11112 \
