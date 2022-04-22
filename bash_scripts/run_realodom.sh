#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe=gym \
    --domain=Locobot \
    --task=RealOdomNav-v0 \
    --exp-name=odom-nav-test \
    --checkpoint-frequency=10 \
    --trial-cpus=8 \
    --trial-gpus=1 \
    --run-eagerly False 
