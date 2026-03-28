#!/bin/bash
#SBATCH -J skyguard_uav
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
echo "SkyGuard SLURM job"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

if [ -f /public/home/zhaohuiqi/.ai_user_info/ai_proxy ]; then
    source /public/home/zhaohuiqi/.ai_user_info/ai_proxy
fi

if [ -f "$HOME/.bashrc" ]; then
    set +u
    source "$HOME/.bashrc"
    set -u
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

PROJECT_CANDIDATES=(
    "$SUBMIT_DIR"
    "$SCRIPT_DIR"
    "/public/home/zhaohuiqi/LUO/FIDSUS-main"
    "/public/home/zhaohuiqi/LUO/FIDSUS-main-source"
)

PROJECT_DIR=""
for candidate in "${PROJECT_CANDIDATES[@]}"; do
    if [ -d "$candidate/system" ] && [ -d "$candidate/dataset" ]; then
        PROJECT_DIR="$candidate"
        break
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    echo "Cannot find project directory."
    echo "Checked paths: ${PROJECT_CANDIDATES[*]}"
    exit 1
fi

if type module >/dev/null 2>&1; then
    MPI_MODULE_CANDIDATES=(
        "mpi/openmpi/openmpi-4.1.5-gcc9.3.0"
        "openmpi/4.1.5"
    )
    for mpi_module in "${MPI_MODULE_CANDIDATES[@]}"; do
        if module load "$mpi_module" >/dev/null 2>&1; then
            echo "Loaded MPI module: $mpi_module"
            break
        fi
    done
fi

unset PYTHONHOME || true
unset PYTHONPATH || true
export CONDA_NO_PLUGINS=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PREFERRED_ENVS=()
GPU_BACKEND="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_BACKEND="nvidia"
    PREFERRED_ENVS=("LUO" "fl")
    echo "Detected NVIDIA GPU node, preferring CUDA-compatible conda envs."
else
    GPU_BACKEND="dcu"
    PREFERRED_ENVS=("pytorch_dtk" "LUO" "fl")
    echo "NVIDIA GPU not detected, preferring DCU-compatible conda envs."
fi

CONDA_SH_CANDIDATES=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
)

CONDA_SH=""
for candidate in "${CONDA_SH_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
    ACTIVATED_ENV=""
    for env_name in "${PREFERRED_ENVS[@]}"; do
        if conda activate "$env_name" >/dev/null 2>&1; then
            ACTIVATED_ENV="$env_name"
            echo "Activated conda env: $env_name"
            break
        fi
    done

    if [ -z "$ACTIVATED_ENV" ]; then
        if [ "$GPU_BACKEND" = "nvidia" ]; then
            echo "No CUDA-compatible conda env found (${PREFERRED_ENVS[*]})."
            echo "Please create or activate a CUDA PyTorch env, then resubmit."
            exit 1
        fi
        echo "No preferred conda env found (${PREFERRED_ENVS[*]}), using current Python environment."
    fi
else
    echo "Conda init script not found, falling back to module/system Python."
    if type module >/dev/null 2>&1; then
        module load python/3.8.10 >/dev/null 2>&1 || true
        echo "Tried loading module: python/3.8.10"
    fi
fi

PYTHON_BIN=""
PYTHON_CANDIDATES=(
    "${CONDA_PREFIX:-}/bin/python"
    "${CONDA_PREFIX:-}/bin/python3"
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

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
    echo "Selected Python cannot import torch: $PYTHON_BIN"
    exit 1
fi

DATASETS=("UAVCAN" "FANET")
ALGORITHMS=("SkyGuardPFIDS")

REQUEST_NUM_CLIENTS="${NUM_CLIENTS:-20}"
JOIN_RATIO="${JOIN_RATIO:-1}"
CLIENT_ACTIVITY_RATE="${CLIENT_ACTIVITY_RATE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-0}"
TIMES="${TIMES:-1}"
MODEL_NAME="${MODEL_NAME:-tcn}"

PUBLIC_VAL_RATIO="${PUBLIC_VAL_RATIO:-0.05}"
PUBLIC_VAL_BATCHES="${PUBLIC_VAL_BATCHES:-2}"
CLIENT_TEST_RATIO="${CLIENT_TEST_RATIO:-0.2}"
NUM_CLUSTERS="${NUM_CLUSTERS:-3}"
POISON_RATIO="${POISON_RATIO:-0.0}"
SYBIL_RATIO="${SYBIL_RATIO:-0.0}"
DROPOUT_RATIO="${DROPOUT_RATIO:-0.0}"
ENERGY_BUDGET="${ENERGY_BUDGET:-0.0}"
PAYMENT_BUDGET="${PAYMENT_BUDGET:-0.0}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

UAVCAN_TASK_MODE="${UAVCAN_TASK_MODE:-binary}"
FANET_TASK_MODE="${FANET_TASK_MODE:-scenario}"
FANET_ATTACK_ANNOTATION="${FANET_ATTACK_ANNOTATION:-}"

SYSTEM_DIR="$PROJECT_DIR/system"
DATASET_DIR="$PROJECT_DIR/dataset"
RUN_ID=$(date +%s)
LOG_BASE_CANDIDATES=(
    "${SLURM_SUBMIT_DIR:-}"
    "$HOME"
    "/tmp/${USER:-zhaohuiqi}"
)

LOG_ROOT=""
for base in "${LOG_BASE_CANDIDATES[@]}"; do
    if [ -n "$base" ] && mkdir -p "$base/fidsus_logs" >/dev/null 2>&1; then
        LOG_ROOT="$base/fidsus_logs/skyguard_run_${RUN_ID}"
        mkdir -p "$LOG_ROOT"
        break
    fi
done

if [ -z "$LOG_ROOT" ]; then
    echo "Cannot create a writable log directory."
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo "Python version: $($PYTHON_BIN --version 2>&1)"
echo "Python path: $PYTHON_BIN"
echo "Fixed run set: only FANET/UAVCAN with SkyGuardPFIDS"
echo "Datasets: ${DATASETS[*]}"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Log root: $LOG_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

dataset_config_field() {
    local config_path="$1"
    local field_name="$2"
    "$PYTHON_BIN" - "$config_path" "$field_name" <<'PY'
import json
import sys

path = sys.argv[1]
field = sys.argv[2]
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)
print(data[field])
PY
}

prepare_uavcan() {
    local output_dir="$DATASET_DIR/UAVCAN"
    local cfg_path="$output_dir/config.json"
    local public_path="$output_dir/public_val.npz"

    if [ "$FORCE_PREPROCESS" != "1" ] && [ -f "$cfg_path" ] && [ -f "$public_path" ]; then
        echo "UAVCAN dataset already prepared: $output_dir"
        return
    fi

    echo "Preparing UAVCAN dataset..."
    "$PYTHON_BIN" "$DATASET_DIR/generate_uavcan.py" \
        --raw-dir "$DATASET_DIR/UAVCAN_Attack_dataset(DroneCAN)" \
        --output-dir "$output_dir" \
        --num-clients "$REQUEST_NUM_CLIENTS" \
        --public-val-ratio "$PUBLIC_VAL_RATIO" \
        --client-test-ratio "$CLIENT_TEST_RATIO" \
        --task-mode "$UAVCAN_TASK_MODE" \
        --num-clusters "$NUM_CLUSTERS"
}

prepare_fanet() {
    local output_dir="$DATASET_DIR/FANET"
    local cfg_path="$output_dir/config.json"
    local public_path="$output_dir/public_val.npz"

    if [ "$FORCE_PREPROCESS" != "1" ] && [ -f "$cfg_path" ] && [ -f "$public_path" ]; then
        echo "FANET dataset already prepared: $output_dir"
        return
    fi

    echo "Preparing FANET dataset..."
    local cmd=(
        "$PYTHON_BIN" "$DATASET_DIR/generate_fanet.py"
        --raw-dir "$DATASET_DIR/FANET_Dataset_NS3.40"
        --output-dir "$output_dir"
        --num-clients "$REQUEST_NUM_CLIENTS"
        --public-val-ratio "$PUBLIC_VAL_RATIO"
        --client-test-ratio "$CLIENT_TEST_RATIO"
        --task-mode "$FANET_TASK_MODE"
        --num-clusters "$NUM_CLUSTERS"
    )
    if [ -n "$FANET_ATTACK_ANNOTATION" ]; then
        cmd+=(--attack-annotation "$FANET_ATTACK_ANNOTATION")
    fi
    "${cmd[@]}"
}

prepare_dataset() {
    local dataset_name="$1"
    case "$dataset_name" in
        UAVCAN)
            prepare_uavcan
            ;;
        FANET)
            prepare_fanet
            ;;
        *)
            echo "No preprocess step registered for dataset: $dataset_name"
            ;;
    esac
}

for DATASET in "${DATASETS[@]}"; do
    prepare_dataset "$DATASET"

    CONFIG_PATH="$DATASET_DIR/${DATASET}/config.json"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Dataset config not found: $CONFIG_PATH, skipping $DATASET"
        continue
    fi

    if ! NUM_CLASSES=$(dataset_config_field "$CONFIG_PATH" "num_classes" 2>/dev/null); then
        echo "Failed to parse num_classes from: $CONFIG_PATH, skipping $DATASET"
        continue
    fi

    if ! DATASET_NUM_CLIENTS=$(dataset_config_field "$CONFIG_PATH" "num_clients" 2>/dev/null); then
        DATASET_NUM_CLIENTS="$REQUEST_NUM_CLIENTS"
    fi

    DATASET_PARTITION="dirichlet"
    case "$DATASET" in
        UAVCAN)
            DATASET_PARTITION="by_session"
            ;;
        FANET)
            DATASET_PARTITION="by_node"
            ;;
    esac

    GOAL_SUFFIX="${DATASET}_nc${DATASET_NUM_CLIENTS}"
    LOG_DIR="$LOG_ROOT/${DATASET}"
    mkdir -p "$LOG_DIR"
    FAILED_LOG="$LOG_DIR/failed_algorithms.txt"
    PREPROCESS_LOG="$LOG_DIR/preprocess_info.txt"

    {
        echo "dataset=$DATASET"
        echo "config=$CONFIG_PATH"
        echo "num_classes=$NUM_CLASSES"
        echo "num_clients=$DATASET_NUM_CLIENTS"
        echo "partition=$DATASET_PARTITION"
        echo "uavcan_task_mode=$UAVCAN_TASK_MODE"
        echo "fanet_task_mode=$FANET_TASK_MODE"
    } > "$PREPROCESS_LOG"

    echo "=========================================="
    echo "Start dataset: $DATASET"
    echo "Num classes: $NUM_CLASSES"
    echo "Num clients: $DATASET_NUM_CLIENTS"
    echo "Logs: $LOG_DIR"
    echo "=========================================="

    cd "$SYSTEM_DIR"
    for ALG in "${ALGORITHMS[@]}"; do
        echo "------------------------------------------"
        echo "Dataset = $DATASET, Algorithm = $ALG, time = $(date)"

        CMD=(
            "$PYTHON_BIN" -u main.py
            -dev "$DEVICE"
            -did "$DEVICE_ID"
            -algo "$ALG"
            -data "$DATASET"
            -go "${ALG}_${GOAL_SUFFIX}"
            -nc "$DATASET_NUM_CLIENTS"
            -nb "$NUM_CLASSES"
            -jr "$JOIN_RATIO"
            -car "$CLIENT_ACTIVITY_RATE"
            -lbs "$BATCH_SIZE"
            -gr "$GLOBAL_ROUNDS"
            -ls "$LOCAL_EPOCHS"
            -m "$MODEL_NAME"
            -t "$TIMES"
            --auto_num_classes true
            --auto_num_clients true
            --client_partition "$DATASET_PARTITION"
            --public_val_ratio "$PUBLIC_VAL_RATIO"
            --public_val_batches "$PUBLIC_VAL_BATCHES"
            --poison_ratio "$POISON_RATIO"
            --sybil_ratio "$SYBIL_RATIO"
            --dropout_ratio "$DROPOUT_RATIO"
            --energy_budget "$ENERGY_BUDGET"
            --payment_budget "$PAYMENT_BUDGET"
            --eval_train_max_batches 2
            --eval_test_max_batches 50
            --eval_calibration_max_batches 2
            --eval_progress_log true
            --skyguard_require_public_val true
            --skyguard_require_real_profiles true
        )

        if "${CMD[@]}" > "$LOG_DIR/${ALG}.log" 2>&1; then
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
