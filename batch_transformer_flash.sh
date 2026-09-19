#! /bin/bash -x
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=72:0:00
#PBS -q capacity
#PBS -A foundmedicine
#PBS -l filesystems=flare
#PBS -k doe
#PBS -N h_sweep_layer_sdpa
#PBS -j oe

source /lus/flare/projects/FoundMedicine/jahatef/setup.sh
cd /lus/flare/projects/FoundMedicine/jahatef/sizing/cookbook/benchmarks/sizing

#python transformer_flops.py --hidden_size 16 32768 16 --num_attention_heads 16 --microbatch_size 4 --seq_length 2048 --vocab_size 51200 --global_batch_size 256 --tensor_mp_size 1 --num_iterations 50 --num_warmup_iterations 25 --blocks qkv_transform attention_score attention_over_value attention_linear_projection mlp_h_to_4h logit_block layer_norm 

HEADS=(8 16 24 32 40 48 56 64 72 80 96 128)

MAX_CONCURRENT=12

for idx in "${!HEADS[@]}"; do
    H=${HEADS[$idx]}

    GPU=$((idx % 12))

    (

        echo "Starting H=$H on GPU $GPU"

        ZE_AFFINITY_MASK=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$((6000 + idx)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python transformer_flops.py \
            --hidden_size $((H * 16)) $((H * 32)) $((H * 64)) $((H * 128)) $((H * 256)) \
            --num_attention_heads "$H" \
            --microbatch_size 4 \
            --seq_length 2048 \
            --vocab_size 51200 \
            --global_batch_size 256 \
            --tensor_mp_size 1 \
            --num_iterations 100 \
            --num_warmup_iterations 50 \
			--blocks flash \
            --output_file "results_h_sweep_flash_median/heads_${H}.csv"


        echo "Finished H=$H"
    ) &

    #
    # Keep at most 6 running simultaneously.
    #
    if (( (idx + 1) % MAX_CONCURRENT == 0 )); then
        wait
    fi
done

wait

echo "All experiments complete."



