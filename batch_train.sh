#!/bin/bash
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -q capacity
#PBS -A foundmedicine
#PBS -l filesystems=flare
#PBS -k doe
#PBS -N neox
#PBS -j oe

#set -Eeuo pipefail
#PBS -l select=1

SCRATCH=/lus/flare/projects/FoundMedicine/jahatef
export NEOX_DIR="$SCRATCH/gpt-neox"

# Distributed configuration: one worker per node.
GPUS_PER_NODE=${1:-12}
GPUS_PER_NODE=12
TOK_PER_RANK=${2:-2048}
NPROC_PER_NODE=$GPUS_PER_NODE

mapfile -t HOSTS < <(sort -u "${PBS_NODEFILE}")
NNODES=${#HOSTS[@]}
RANKS=$((NNODES * GPUS_PER_NODE))

export TOTAL_TOK=$((TOK_PER_RANK * RANKS))


export TMPDIR=/tmp
export NUMEXPR_NUM_THREADS=64

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MASTER_ADDR="${HOSTS[0]}"
export MASTER_PORT=$((29500 + ${PBS_JOBID%%.*} % 1000))
export NNODES NPROC_PER_NODE
export DEVICE_TYPE=xpu
export CCL_PROCESS_LAUNCHER=torchrun
export CCL_ATL_TRANSPORT=mpi
#export ZE_AFFINITY_MASK=0,1
export BACKEND=fused
export WANDB_MODE=offline
export PYTHONPATH="$NEOX_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_AUTOTUNE_CACHE_DIR="$SCRATCH/triton_cache/autotune"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache/cache"
export TRITON_PRINT_AUTOTUNING=1
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export LOCAL_SIZE=$GPUS_PER_NODE
echo $WORLD_SIZE
echo $LOCAL_SIZE

source "$SCRATCH/setup.sh"

# Preserve the hostfile if the local DeepSpeed/deepy launcher uses it.
HOSTFILE_DIR="$SCRATCH/hostfiles"
mkdir -p "$HOSTFILE_DIR" "$SCRATCH/logs"
HOSTFILE="$HOSTFILE_DIR/hosts_${PBS_JOBID%%.*}"

for host in "${HOSTS[@]}"; do
    echo "$host slots=$GPUS_PER_NODE"
done > "$HOSTFILE"

cat $HOSTFILE
export DLTS_HOSTFILE="$HOSTFILE"

cd "$NEOX_DIR"

CONFIG_DIR="$SCRATCH/sizing/scripts/configs/${RANKS}_ranks"
SETUP_CONFIG="$NEOX_DIR/configs/local_setup_wandb.yml"
LOG_DIR="$SCRATCH/sizing/scripts/logs/${RANKS}_ranks"

mkdir -p "$LOG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "Missing config directory: $CONFIG_DIR" >&2; exit 1; }
[[ -f "$SETUP_CONFIG" ]] || { echo "Missing setup config: $SETUP_CONFIG" >&2; exit 1; }

mapfile -t CONFIGS < <(
    find "$CONFIG_DIR" -maxdepth 1 -type f \
        \( -name '*.yml' -o -name '*.yaml' \) | sort
)

((${#CONFIGS[@]})) || { echo "No YAML configs in $CONFIG_DIR" >&2; exit 1; }

echo "Nodes: ${HOSTS[*]}"
echo "World size: $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Configs: $CONFIG_DIR"
echo "Logs: $LOG_DIR"

#for path in "${CONFIGS[@]}"; do
#config=${path##*/}
config="$SCRATCH/sizing/scripts/configs/1-3B.yml"
#config="$SCRATCH/sizing/scripts/configs/6-7B.yml"
#config="$SCRATCH/sizing/scripts/configs/2-7B.yml"
log="$LOG_DIR/$RANKS.log"

echo "[$(date)] Starting $config"

# Start each experiment without the previous experiment's checkpoints.
rm -rf checkpoints/latest checkpoints/global_step*

python deepy.py train.py \
	-d "$CONFIG_DIR" \
	"$config" \
	"$SCRATCH/sizing/scripts/configs/cp-yamls/cp${RANKS}.yml" \
	"$SCRATCH/sizing/scripts/configs/seq_configs/${TOTAL_TOK}.yml" \
	"$SETUP_CONFIG" 2>&1 | tee "$log"

echo "[$(date)] Finished $config"
#done


