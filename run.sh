#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=slurm_logs/%j/out.txt
#SBATCH --error=slurm_logs/%j/err.txt
#SBATCH --nodes=1
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00

PYTHON=/users/sbiswas/miniforge3/envs/sgl/bin/python


# Load your environment (e.g., anaconda, modules, virtualenv)
# source activate your_env_name

# Base arguments that don't change
MODEL="simple-cnn"
DATASET="cifar10"
ALG="fedmuon"
BATCH_SIZE=64
EPOCHS=10
N_PARTIES=10
RHO=0.9
COMM_ROUND=100
BETA=0.5
DATADIR="./data/"
LOGDIR="./logs/"
NOISE=0
LR=0.01
PARTITION="noniid-#label2"
K=1000

# Array of learning rates or seeds you might want to vary (example setup)
# You can change these arrays to configure your 4 distinct parameters
LRS=(0.01 0.01 0.05 0.05)
SEEDS=(0 1 2 3)
# KS=(50 100 500 1000)
GAMMAS=(0.3 0.5 0.7 0.9)
# Run 4 experiments in parallel (one on each GPU)
for i in {0..3}
do
    # Calculate the specific GPU index
    GPU_INDEX=$i
    
    echo "Starting Experiment $i on GPU $GPU_INDEX with..."
    
    $PYTHON experiments.py \
        --model=$MODEL \
        --dataset=$DATASET \
        --alg=$ALG \
        --lr=$LR \
        --batch-size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        --n_parties=$N_PARTIES \
        --rho=$RHO \
        --comm_round=$COMM_ROUND \
        --partition=$PARTITION \
        --beta=$BETA \
        --device="cuda:$GPU_INDEX" \
        --datadir=$DATADIR \
        --logdir=$LOGDIR \
        --noise=$NOISE \
        --K=$K \
        --gamma=${GAMMAS[$i]} \
        --init_seed=0 & 

    sleep 61
    
done

wait
echo "All 4 experiments have completed!"