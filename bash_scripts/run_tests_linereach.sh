#!/bin/bash
export RAY_MEMORY_MONITOR_ERROR_THRESHOLD=1.000001

softlearning run_example_local examples.development \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Tests \
    --task LineReach-v0 \
    --exp-name tests-line-reach-test \
    --checkpoint-frequency 10 \
    --gpus 10 \
    --trial-cpus 1 \
    --trial-gpus 0.45 \
    --max-failures 0 \
    --run-eagerly False \
    --server-port 11101 \
    --temp-dir ~/tmp/ \
