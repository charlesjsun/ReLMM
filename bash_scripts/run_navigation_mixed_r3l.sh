#!/bin/bash
softlearning run_example_local examples.r3l \
    --algorithm R3L \
    --universe gym \
    --domain Locobot \
    --task MixedNavigationResetFree-v0 \
    --exp-name locobot-mixed-navigation-reset-free-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 17777 \
    --num-samples 1 \