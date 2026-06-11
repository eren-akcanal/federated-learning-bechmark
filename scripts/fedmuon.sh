#!/usr/bin/env bash
# =============================================================================
# sweep_fedmuon.sh
#
# Full hyperparameter sweep for FedMuon across:
#   4 µ values        : 0.001  0.01  0.1  1.0
#   6 datasets        : mnist  fmnist  cifar10  a9a  rcv1  covtype
#   6 partitioning    : pk~Dir(0.5)   #C=1  #C=2  #C=3
#                       x̂~Gau(0.1)   q~Dir(0.5)
#
# Total combinations : 4 × 6 × 6 = 144 runs
#
# Usage:
#   chmod +x sweep_fedmuon.sh
#   ./sweep_fedmuon.sh                        # run jobs in parallel (max 4)
#   ./sweep_fedmuon.sh 2>&1 | tee sweep.log   # same, with a log file
#
# Optional overrides via environment variables:
#   CONCURRENT_JOBS=4 ./sweep_fedmuon.sh       # change max parallel processes
#   DEVICE=cuda:1     ./sweep_fedmuon.sh       # use a different GPU
#   DRY_RUN=1         ./sweep_fedmuon.sh       # print commands, don't run them
#   WORKERS=4         ./sweep_fedmuon.sh       # dataloader threads (default 6)
#   SKIP_DONE=1       ./sweep_fedmuon.sh       # skip if log file already exists
# =============================================================================

set -euo pipefail

# ── tuneable defaults ────────────────────────────────────────────────────────
CONCURRENT_JOBS="${CONCURRENT_JOBS:-4}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${WORKERS:-6}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-0}"

DATADIR="./data/"
LOGDIR="./logs/"
MODELDIR="./models/"

# Fixed FedMuon hyper-parameters (match your example command)
COMM_ROUND=50
N_PARTIES=10
BATCH_SIZE=64
EPOCHS=10
RHO=0.9
SAMPLE=1
INIT_SEED=0
NOISE=0
OPTIMIZER="sgd"

# ── sweep axes ───────────────────────────────────────────────────────────────

MU_VALUES=(0.001 0.01 0.1 1.0)

# Each dataset entry: "dataset_flag  model_flag  lr"
#   - Image datasets  → simple-cnn, lr=0.01
#   - Tabular datasets → mlp,        lr=0.01
declare -A DS_MODEL=(
    [mnist]="simple-cnn"
    [fmnist]="simple-cnn"
    [cifar10]="simple-cnn"
    [a9a]="mlp"
    [rcv1]="mlp"
    [covtype]="mlp"
)
declare -A DS_LR=(
    [mnist]="0.01"
    [fmnist]="0.01"
    [cifar10]="0.01"
    [a9a]="0.01"
    [rcv1]="0.01"
    [covtype]="0.01"
)
DATASETS=(mnist fmnist cifar10 a9a rcv1 covtype)

# Partition strategies — each entry is a unique key; the run_experiment
# function below maps it to the correct --partition / --beta / --noise flags.
#
#   pk~Dir(0.5)   → label-distribution skew   (noniid-labeldir,  beta=0.5)
#   #C=1          → 1 class per client         (noniid-#label1)
#   #C=2          → 2 classes per client       (noniid-#label2)
#   #C=3          → 3 classes per client       (noniid-#label3)
#   x̂~Gau(0.1)   → feature-space noise skew   (homo + noise_type=space, noise=0.1)
#   q~Dir(0.5)    → quantity skew              (hetero-dir,        beta=0.5)
#
PARTITIONS=(pk_dir0.5 C1 C2 C3 gau0.1 q_dir0.5)

# ── helpers ──────────────────────────────────────────────────────────────────

total_runs=$(( ${#MU_VALUES[@]} * ${#DATASETS[@]} * ${#PARTITIONS[@]} ))
run_idx=0

run_experiment() {
    local mu="$1"
    local dataset="$2"
    local partition_key="$3"

    local model="${DS_MODEL[$dataset]}"
    local lr="${DS_LR[$dataset]}"

    # Map partition key → CLI flags
    local partition_flag beta_flag noise_flag noise_type_flag
    noise_flag="--noise 0"
    noise_type_flag="--noise_type level"

    case "$partition_key" in
        pk_dir0.5)
            partition_flag="--partition noniid-labeldir"
            beta_flag="--beta 0.5"
            ;;
        C1)
            partition_flag="--partition noniid-#label1"
            beta_flag="--beta 0.5"   # beta unused for #label but kept for logging
            ;;
        C2)
            partition_flag="--partition noniid-#label2"
            beta_flag="--beta 0.5"
            ;;
        C3)
            partition_flag="--partition noniid-#label3"
            beta_flag="--beta 0.5"
            ;;
        gau0.1)
            # Feature-space Gaussian noise skew: IID splits + additive noise
            partition_flag="--partition homo"
            beta_flag="--beta 0.5"
            noise_flag="--noise 0.1"
            noise_type_flag="--noise_type space"
            ;;
        q_dir0.5)
            partition_flag="--partition hetero-dir"
            beta_flag="--beta 0.5"
            ;;
        *)
            echo "ERROR: unknown partition key '$partition_key'" >&2
            exit 1
            ;;
    esac

    # Build a readable log-file name so each run has its own file
    local log_name="fedmuon_${dataset}_${partition_key}_mu${mu}"
    local log_file="${LOGDIR}${log_name}.log"

    run_idx=$(( run_idx + 1 ))
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " Dispatching Run ${run_idx}/${total_runs}"
    echo "   alg=fedmuon  dataset=${dataset}  partition=${partition_key}  mu=${mu}"
    echo "   model=${model}  lr=${lr}"
    echo "   log → ${log_file}"
    echo "═══════════════════════════════════════════════════════════════"

    # Skip if log already exists and SKIP_DONE=1
    if [[ "$SKIP_DONE" == "1" && -f "$log_file" ]]; then
        echo "  [SKIP] log file exists, skipping."
        return
    fi

    local cmd=(
        python experiments.py
            --model="$model"
            --dataset="$dataset"
            --alg=fedmuon
            --lr="$lr"
            --batch-size="$BATCH_SIZE"
            --epochs="$EPOCHS"
            --n_parties="$N_PARTIES"
            --mu="$mu"
            --rho="$RHO"
            --comm_round="$COMM_ROUND"
            $partition_flag
            $beta_flag
            --device="$DEVICE"
            --datadir="$DATADIR"
            --logdir="$LOGDIR"
            --modeldir="$MODELDIR"
            $noise_flag
            $noise_type_flag
            --sample="$SAMPLE"
            --init_seed="$INIT_SEED"
            --optimizer="$OPTIMIZER"
            --num_workers="$NUM_WORKERS"
            --log_file_name="$log_name"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY RUN] ${cmd[*]}"
    else
        # Launch python process in the background
        "${cmd[@]}" &
        
        # If active background processes hit our limit, halt loop until one finishes
        if (( $(jobs -r -p | wc -l) >= CONCURRENT_JOBS )); then
            wait -n
        fi
    fi
}

# ── main sweep loop ──────────────────────────────────────────────────────────

echo "Starting FedMuon parallel sweep: ${total_runs} total jobs"
echo "  Max Concurrency: ${CONCURRENT_JOBS}"
echo "  Device         : ${DEVICE}"
echo "  Workers        : ${NUM_WORKERS}"
echo "  Dry-run        : ${DRY_RUN}"
echo "  Skip done      : ${SKIP_DONE}"
echo ""

mkdir -p "$LOGDIR" "$MODELDIR"

for mu in "${MU_VALUES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for partition in "${PARTITIONS[@]}"; do
            run_experiment "$mu" "$dataset" "$partition"
        done
    done
done

# Wait for the remaining running jobs to complete
echo "All jobs dispatched. Waiting for the final batch to finish..."
wait

echo ""
echo "Sweep complete. ${run_idx}/${total_runs} jobs processed."