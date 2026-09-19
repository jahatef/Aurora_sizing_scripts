#! /bin/bash +x
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=72:0:00
#PBS -q capacity
#PBS -A foundmedicine
#PBS -l filesystems=flare
#PBS -k doe
#PBS -N s_sweep
#PBS -j oe

source /lus/flare/projects/FoundMedicine/jahatef/setup.sh
cd /lus/flare/projects/FoundMedicine/jahatef/sizing/cookbook/benchmarks/sizing

#python transformer_flops.py --hidden_size_range 16 32768 16 --num_attention_heads 16 --microbatch_size 4 --seq_length 2048 --vocab_size 51200 --global_batch_size 256 --tensor_mp_size 1 --num_iterations 50 --num_warmup_iterations 25 --blocks qkv_transform attention_score attention_over_value attention_linear_projection mlp_h_to_4h logit_block layer_norm 

SEQLENS=(2048 4096 8192 16384 32768 65536 131072) 	

MAX_CONCURRENT=12

for idx in "${!SEQLENS[@]}"; do
    S=${SEQLENS[$idx]}

    GPU=$((idx % 12 * 2))

    (

        echo "Starting S=$S on GPU $GPU"

        ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size_range 32 16384 32 \
            --num_attention_heads 32 \
            --microbatch_size 1 \
            --seq_length "$S" \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 100 \
            --num_warmup_iterations 50 \
			--blocks attention_score \
			         attention_over_value \
                     qkv_transform \
                     attention_score \
                     attention_over_value \
                     attention_linear_projection \
                     mlp_h_to_4h \
                     mlp_4h_to_h \
                     logit_block \
                     sdpa \
            --output_file "results_s_sweep_blocks_median/seqlen_${S}.csv"

        'ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size_range 32 16384 32 \
            --num_attention_heads 32 \
            --microbatch_size 4 \
            --seq_length "$S" \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 50 \
            --num_warmup_iterations 25 \
			--blocks attention_score \
            --output_file "results_s_sweep_gemms/seqlen_${S}.csv"

        ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size_range 32 16384 32 \
            --num_attention_heads 32 \
            --microbatch_size 4 \
            --seq_length "$S" \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 50 \
            --num_warmup_iterations 25 \
			--blocks attention_over_value \
            --output_file "results_s_sweep_gemms/seqlen_${S}.csv"

        ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size_range 32 16384 32 \
            --num_attention_heads 32 \
            --microbatch_size 4 \
            --seq_length "$S" \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 50 \
            --num_warmup_iterations 25 \
			--blocks attention_linear_projection \
            --output_file "results_s_sweep_gemms/seqlen_${S}.csv"

        ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size_range 32 16384 32 \
            --num_attention_heads 32 \
            --microbatch_size 4 \
            --seq_length "$S" \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 50 \
            --num_warmup_iterations 25 \
			--blocks mlp_h_to_4h \
            --output_file "results_s_sweep_gemms/seqlen_${S}.csv"

        ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size_range 32 16384 32 \
            --num_attention_heads 32 \
            --microbatch_size 4 \
            --seq_length "$S" \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 50 \
            --num_warmup_iterations 25 \
			--blocks mlp_4h_to_h \
            --output_file "results_s_sweep_gemms/seqlen_${S}.csv"

        ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size_range 32 16384 32 \
            --num_attention_heads 32 \
            --microbatch_size 4 \
            --seq_length "$S" \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 50 \
            --num_warmup_iterations 25 \
			--blocks logit_block \
            --output_file "results_s_sweep_gemms/seqlen_${S}.csv"
            '

        echo "Finished S=$S"
    ) &

    #
			#--blocks qkv_transform attention_score attention_over_value attention_linear_projection mlp_h_to_4h mlp_4h_to_h logit_block \
			#--blocks qkv_transform attention_score attention_over_value attention_linear_projection mlp_h_to_4h  mlp_4h_to_h logit_block\
			#--blocks qkv_transform attention_score attention_over_value attention_linear_projection mlp_h_to_4h mlp_4h_to_h logit_block sweep_attn \
    # Keep at most 6 running simultaneously.
    #
    if (( (idx + 1) % MAX_CONCURRENT == 0 )); then
        wait
    fi
done

wait

echo "All experiments complete."


#ZE_AFFINITY_MASK=0  MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + 1)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py --hidden_size_range_range 512 16384 512 --num_attention_heads 32 --microbatch_size 4 --seq_length 16384 --vocab_size 51200 --global_batch_size 256 --tensor_mp_size 1 --num_iterations 50 --num_warmup_iterations 25 --blocks qkv_transform attention_score attention_over_value attention_linear_projection mlp_h_to_4h mlp_4h_to_h logit_block sweep_attn --output_file "results_attention/seqlen_test.csv" 



