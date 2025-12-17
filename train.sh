#!/bin/bash

# Check if the number of cores is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_cores>"
    exit 1
fi

NUM_CORES=$1

srun -AACD114118 \
    -N1 -n1 \
    -c"$NUM_CORES" \
    ./train_2048_openmp 40000 1000 22500 log_"$NUM_CORES".json
