#!/bin/bash
softlearning run_example_local examples.real \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task RealNavigationRND-v0 \
    --exp-name real-navigation-rnd-test \
    --checkpoint-frequency 5 \
    --trial-cpus 16 \
    --trial-gpus 0 \
    --run-eagerly False \
    --server-port 12222 \
