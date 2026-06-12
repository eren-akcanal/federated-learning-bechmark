#!/usr/bin/env bash
set -euo pipefail

# Maximum number of python jobs running at the exact same time

PARTITIONS=(
    "iid-diff-quantity" 
    "homo"
)
DATASETS=("rcv1" "a9a" "covtype")

echo "Starting execution queue..."

for partition in "${PARTITIONS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        
        echo "Launching: dataset=$dataset, partition=$partition"

        log_file="log_${dataset}_${partition}"
        
        # Run in background via &
        python experiments.py --model=mlp \
            --dataset="$dataset" \
            --alg=fedsophia \
            --lr=0.01 \
            --batch-size=64 \
            --epochs=10 \
            --n_parties=10 \
            --rho=0.9 \
            --mu=0.01 \
            --comm_round=50 \
            --partition="$partition" \
            --beta=0.5 \
            --device='cuda:0' \
            --datadir='./data/' \
            --logdir='./fedmuon/' \
            --noise=0 \
            --sample=1 \
            --init_seed=0 \
            --optimizer muon


    done
done

# Block until the very last batch of straggler processes finishes up
echo "All experiments dispatched. Waiting for final jobs to complete..."
wait
echo "All tasks finished successfully!"