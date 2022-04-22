#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationVacuumResetFree-v0 \
    --exp-name locobot-navigation-vacuum-reset-free-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 1 \
    --run-eagerly False \
    --server-port 11112 \
