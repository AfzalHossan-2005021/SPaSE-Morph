#!/bin/bash

alpha_options=(0.0001 0.001 0.01 0.1)
lambda_sinkhorn_options=(0.001 0.01 0.1)
# alpha_options=(0.1)
# lambda_sinkhorn_options=(0.1)

# Python file to execute
python_file="run.py"

# Loop over all combinations of alpha and lambda_sinkhorn
for alpha in "${alpha_options[@]}"; do
  for lambda_sinkhorn in "${lambda_sinkhorn_options[@]}"; do
    echo "Running $python_file with alpha=$alpha and lambda_sinkhorn=$lambda_sinkhorn"
    python $python_file --alpha $alpha --lambda_sinkhorn $lambda_sinkhorn
  done
done