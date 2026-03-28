#!/bin/bash
#SBATCH -J fidsus_multi
#SBATCH -p ksagexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -o slurm_out.%j
#SBATCH -e slurm_err.%j
#SBATCH --time=48:00:00

set -uo pipefail

echo "=========================================="
echo "FIDSUS multi-dataset SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

if [ -f /public/home/zhaohuiqi/.ai_user_info/ai_proxy ]; then
    source /public/home/zhaohuiqi/.ai_user_info/ai_proxy
fi

PROJECT_DIR="/public/home/zhaohuiqi/LUO/FIDSUS-main-source"
if [ ! -d "$PROJECT_DIR/system" ] || [ ! -d "$PROJECT_DIR/dataset" ]; then
    echo "Invalid project directory: $PROJECT_DIR"
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    if conda env list | grep -q '^fl '; then
        conda activate fl || true
        echo "Activated conda env: fl"
    else
        echo "Conda env 'fl' not found, using current Python environment."
    fi
else
    echo "Conda not found, using current Python environment."
fi

if type module >/dev/null 2>&1; then
    module load python/3.8.10 >/dev/null 2>&1 || true
    echo "Tried loading module: python/3.8.10"
fi

PYTHON_BIN=""
PYTHON_CANDIDATES=(
    "/opt/conda/bin/python3"
    "/opt/conda/envs/fl/bin/python"
    "/usr/bin/python3"
    "/usr/local/bin/python3"
    "python3"
    "python3.12"
    "python3.11"
    "python3.10"
    "python3.9"
    "/opt/conda/bin/python"
    "python"
)

echo "Checking Python candidates..."
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    resolved=""
    if [ -x "$candidate" ]; then
        resolved="$candidate"
    elif command -v "$candidate" >/dev/null 2>&1; then
        resolved=$(command -v "$candidate")
    fi

    if [ -n "$resolved" ]; then
        version_output=$($resolved --version 2>&1)
        echo "Found candidate: $resolved ($version_output)"
        if "$resolved" -c "import sys; sys.exit(0 if sys.version_info[0] >= 3 else 1)"; then
            PYTHON_BIN="$resolved"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "No usable Python 3 interpreter found."
    echo "Tried candidates: ${PYTHON_CANDIDATES[*]}"
    exit 1
fi

DATASETS=("UNSW" "NSLKDD")
NUM_CLIENTS=50
JOIN_RATIO=1
CLIENT_ACTIVITY_RATE=1.0
BATCH_SIZE=64
DEVICE="cuda"
DEVICE_ID=0
TIMES=1
ALGORITHMS=("HGFIDSUS")

SYSTEM_DIR="$PROJECT_DIR/system"
RUN_ID=$(date +%s)
LOG_BASE_CANDIDATES=(
    "${SLURM_SUBMIT_DIR:-}"
    "$HOME"
    "/tmp/${USER:-zhaohuiqi}"
)

LOG_ROOT=""
for base in "${LOG_BASE_CANDIDATES[@]}"; do
    if [ -z "$base" ]; then
        continue
    fi

    if mkdir -p "$base/fidsus_logs" >/dev/null 2>&1; then
        CANDIDATE_ROOT="$base/fidsus_logs/multi_dataset_${RUN_ID}"
        if mkdir -p "$CANDIDATE_ROOT" >/dev/null 2>&1 && [ -w "$CANDIDATE_ROOT" ]; then
            LOG_ROOT="$CANDIDATE_ROOT"
            break
        fi
    fi
done

if [ -z "$LOG_ROOT" ]; then
    echo "Cannot create a writable log directory."
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo "Python version: $($PYTHON_BIN --version 2>&1)"
echo "Python path: $PYTHON_BIN"
echo "Log root: $LOG_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

for DATASET in "${DATASETS[@]}"; do
    CONFIG_PATH="$PROJECT_DIR/dataset/${DATASET}/config.json"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Dataset config not found: $CONFIG_PATH, skipping $DATASET"
        continue
    fi

    if ! NUM_CLASSES=$($PYTHON_BIN -c "import json, pathlib, sys; config_path = pathlib.Path(r'${CONFIG_PATH}'); data = json.loads(config_path.read_text(encoding='utf-8')); print(data['num_classes'])" 2>/dev/null); then
        echo "Invalid dataset config: $CONFIG_PATH, skipping $DATASET"
        continue
    fi
    GOAL_SUFFIX="${DATASET}_nc${NUM_CLIENTS}"
    LOG_DIR="$LOG_ROOT/${DATASET}"
    mkdir -p "$LOG_DIR"
    FAILED_LOG="$LOG_DIR/failed_algorithms.txt"

    echo "=========================================="
    echo "Start dataset: $DATASET"
    echo "Num classes: $NUM_CLASSES"
    echo "Logs: $LOG_DIR"
    echo "=========================================="

    cd "$SYSTEM_DIR"
    for ALG in "${ALGORITHMS[@]}"; do
        echo "------------------------------------------"
        echo "Dataset = $DATASET, Algorithm = $ALG, time = $(date)"

        if "$PYTHON_BIN" -u main.py \
            -dev "$DEVICE" \
            -did "$DEVICE_ID" \
            -algo "$ALG" \
            -data "$DATASET" \
            -go "${ALG}_${GOAL_SUFFIX}" \
            -nc "$NUM_CLIENTS" \
            -nb "$NUM_CLASSES" \
            -jr "$JOIN_RATIO" \
            -car "$CLIENT_ACTIVITY_RATE" \
            -lbs "$BATCH_SIZE" \
            -t "$TIMES" \
            > "$LOG_DIR/${ALG}.log" 2>&1; then
            echo "Completed $ALG at $(date)"
        else
            EXIT_CODE=$?
            echo "Failed $ALG at $(date), exit code = $EXIT_CODE"
            echo "$ALG exit_code=$EXIT_CODE time=$(date '+%F %T')" >> "$FAILED_LOG"
            continue
        fi
    done

    cd "$PROJECT_DIR"
    echo "Finished dataset $DATASET"
done

echo "=========================================="
echo "All datasets finished at $(date)"
echo "Root logs: $LOG_ROOT"
echo "=========================================="
