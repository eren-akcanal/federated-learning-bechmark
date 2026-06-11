#!/bin/bash

# --- Experiment Grid Configurations ---
TRIALS="0 1 2"

# Mapped to match standard script arguments for NIID-Bench style repositories:
# pk~Dir(0.5) -> noniid-labeldir (with beta=0.5)
# #C=1,2,3    -> noniid-#label1, noniid-#label2, noniid-#label3
# x̂~Gau(0.1)  -> noniid-featuredir (or feature_skew)
# q~Dir(0.5)   -> iid-diff-quantity (or homo-skew)
PARTITIONS="noniid-labeldir noniid-#label1 noniid-#label2 noniid-#label3 feature_skew iid-diff-quantity"

DATASETS="mnist fmnist cifar10 svhn fcube femnist adult rcv1 covtype"
MUS="0.001 0.01 0.1 1"

# --- Main Grid Search Loop ---
for init_seed in $TRIALS
do
    for partition in $PARTITIONS
    do
        # Explicitly handling hyperparameter variables linked to strategies
        beta=0.5
        if [ "$partition" == "noniid-labeldir" ]; then
            beta=0.5
        elif [ "$partition" == "iid-diff-quantity" ]; then
            beta=0.5 # or custom quantity skew dirichlet alpha if required
        fi

        for dataset in $DATASETS
        do
            # Automatically adjusting model architectures based on dataset complexity
            model="mlp"
            if [ "$dataset" == "cifar10" ] || [ "$dataset" == "svhn" ] || [ "$dataset" == "femnist" ]; then
                model="simple-cnn" # Or 'resnet' depending on your file's supported arguments
            fi

            for mu in $MUS
            do
                echo "========================================================================"
                echo "Running: FedSAP/FedSOAP | Seed: $init_seed | Partition: $partition | Dataset: $dataset | Mu: $mu"
                echo "========================================================================"
                
                # Execution command
                python experiments.py \
                    --model=$model \
                    --dataset=$dataset \
                    --alg=fedsoap \
                    --lr=0.01 \
                    --batch-size=64 \
                    --epochs=10 \
                    --n_parties=10 \
                    --rho=0.9 \
                    --mu=$mu \
                    --comm_round=50 \
                    --partition=$partition \
                    --beta=$beta \
                    --device='cuda:0' \
                    --datadir='./data/' \
                    --logdir='./logs/' \
                    --noise=0 \
                    --init_seed=$init_seed

                # Optional: Catch failing run crashes early
                if [ $? -ne 0 ]; then
                    echo "Run failed! Exiting script to prevent log pollution."
                    exit 1
                fi
            done
        done
    done
done