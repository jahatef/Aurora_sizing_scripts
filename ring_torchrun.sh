#! /bin/bash -x

seq_len=$1
nodes=$2
ppn=$3


export DEVICE_TYPE=xpu
#export CCL_PROCESS_LAUNCHER=torchrun
export TRITON_PRINT_AUTOTUNING="1"
export RANK=$PMIX_RANK
export LOCAL_RANK=$PALS_LOCAL_RANKID
export WORLD_SIZE=$((ppn * nodes))
export TRITON_CACHE_DIR=/lus/flare/projects/FoundMedicine/jahatef/triton_cache/cache
export TRITON_AUTOTUNE_CACHE_DIR=/lus/flare/projects/FoundMedicine/jahatef/triton_cache/autotune 
  

echo "local_rank: ${LOCAL_RANK}"
echo "rank is ${RANK}"
   
#RINX_ATTN_BACKEND=fused \
#python ${REPO_DIR}/benchmark/benchmark_ringX_func.py \
#--module "ringX_attn.ringX4_attn" \
#--impl "ringX4_attn_func" \
#--batch_size 4 \
#--num_iter 100 \
#--num_warmup 100 \
#--seq_length $seq_len \
#--num_heads 32 \
#--head_dim 64 \
#--dtype "bfloat16" \
#--modes "fwd_bwd" \
#--causal
#--seq_length 196608 \

#torchrun \
#--nproc_per_node=12 \
RINGX_ATTN_BACKEND=fused \
python ${REPO_DIR}/benchmark/benchmark_ringX_func.py \
--module "ringX_attn.ringX1_attn" \
--impl "ringX1_attn_func" \
--batch_size 4 \
--num_iter 100 \
--num_warmup 100 \
--seq_length $seq_len \
--num_heads 32 \
--head_dim 64 \
--dtype "bfloat16" \
--modes "fwd_bwd" \
--causal
		

RINGX_ATTN_BACKEND=fused \
python ${REPO_DIR}/benchmark/benchmark_ringX_func.py \
--module "ringX_attn.ringX2_attn" \
--impl "ringX2_attn_func" \
--batch_size 4 \
--num_iter 100 \
--num_warmup 100 \
--seq_length $seq_len \
--num_heads 32 \
--head_dim 64 \
--dtype "bfloat16" \
--modes "fwd_bwd" \
--causal

RINGX_ATTN_BACKEND=fused \
python ${REPO_DIR}/benchmark/benchmark_ringX_func.py \
--module "ringX_attn.ringX3_attn" \
--impl "ringX3_attn_func" \
--batch_size 4 \
--num_iter 100 \
--num_warmup 100 \
--seq_length $seq_len \
--num_heads 32 \
--head_dim 64 \
--dtype "bfloat16" \
--modes "fwd_bwd" \
--causal

RINGX_ATTN_BACKEND=portable \
python ${REPO_DIR}/benchmark/benchmark_ringX_func.py \
--module "ringX_attn.ringX4_attn" \
--impl "ringX4_attn_func" \
--batch_size 4 \
--num_iter 100 \
--num_warmup 100 \
--seq_length $seq_len \
--num_heads 32 \
--head_dim 64 \
--dtype "bfloat16" \
--modes "fwd_bwd" \
--causal \
