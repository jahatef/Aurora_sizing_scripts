#! /bin/bash -x
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=4:0:00
#PBS -q capacity
#PBS -A foundmedicine
#PBS -l filesystems=flare
#PBS -k doe
#PBS -N 1-node-12ppn-ring
#PBS -j oe

PPN=12
NODES=1
TOTAL_RANKS=$((PPN * NODES))
WEAK_SEQLENS=("$((TOTAL_RANKS * 1024))" "$((TOTAL_RANKS * 2048))" "$((TOTAL_RANKS * 4096))")
TOK_PER_GPUS=(1024 2048 4096)
STRONG_SEQLEN=24576

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
#export ZES_ENABLE_SYSMAN=1
source /lus/flare/projects/FoundMedicine/jahatef/setup.sh
export REPO_DIR=/lus/flare/projects/FoundMedicine/jahatef/ringX-attention-alcf
export TRITON_CACHE_DIR=/lus/flare/projects/FoundMedicine/jahatef/triton_cache/cache
export TRITON_AUTOTUNE_CACHE_DIR=/lus/flare/projects/FoundMedicine/jahatef/triton_cache/autotune 
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}
ROOT=/lus/flare/projects/FoundMedicine/jahatef
 
export DEVICE_TYPE=xpu
  
   
export TRITON_PRINT_AUTOTUNING="1"

#!/bin/bash
set -e



###############################################################################
# Strong scaling
###############################################################################
#for RANKS in 12 6 3 2 1; do
#RANK=$PMI_RANK LOCAL_RANK=$MPI_LOCAL_RANKID WORLD_SIZE=$PMI_SIZE MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + RANK))  \

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29570

cd /lus/flare/projects/FoundMedicine/jahatef/sizing/scripts/
#weak
for TOK_PER_GPU in "${TOK_PER_GPUS[@]}"; do
	LOGDIR=$ROOT/sizing/scripts/ring_logs/${TOK_PER_GPU}_tokens_per_rank
    WEAK_SEQLEN=$((TOK_PER_GPU * TOTAL_RANKS))
	mkdir -p $LOGDIR/weak
	mpirun --envall -n $TOTAL_RANKS --ppn $PPN ./ring_torchrun.sh $WEAK_SEQLEN $NODES $PPN | tee $LOGDIR/weak/${TOTAL_RANKS}_ranks
done
#stron
#mpirun --envall -n $TOTAL_RANKS --ppn $PPN ./ring_torchrun.sh $STRONG_SEQLEN $NODES $PPN | tee $LOGDIR/strong/${TOTAL_RANKS}_ranks

