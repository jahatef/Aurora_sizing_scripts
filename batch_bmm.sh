#! /bin/bash -x
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q capacity
#PBS -A foundmedicine
#PBS -l filesystems=flare
#PBS -k doe
#PBS -N bmm
#PBS -j oe

source /lus/flare/projects/FoundMedicine/jahatef/setup.sh
cd /lus/flare/projects/FoundMedicine/jahatef/sizing/cookbook/benchmarks/sizing

#!/bin/bash
# Rewrites the bench_list / benchmark_bmm / benchmark_bmm_min experiment
# list using bmm_flops.py. Unlike mm_flops.py, bmm_flops.py has a native
# -b/--b_range flag, so no batch-folding is needed here -- b is passed
# through directly.
#
# NOTE on paired sweeps: --m_range/--n_range/--k_range (and -b/-m/-n/-k
# lists) are independent dimensions that get combined, NOT zipped 1:1.
# The "non-square MMs" experiment sweeps outer_dim with m=k=outer_dim
# (a fixed relationship, not independent), so that one is expressed as a
# bash loop issuing one bmm_flops.py call per outer_dim value (each call
# still covers the full -b list of batch sizes in one invocation, since
# b is independent of outer_dim).

mkdir -p results

MAX_CONCURRENT=6
job_idx=0

# Rolling GPU-slot pool: SLOT_PID[i] holds the PID currently occupying
# GPU slot i (0..MAX_CONCURRENT-1), or empty if the slot is free. A new
# job is launched the instant a slot frees up, rather than waiting for
# an entire batch of 6 to finish.
declare -a SLOT_PID
for ((i = 0; i < MAX_CONCURRENT; i++)); do
    SLOT_PID[$i]=""
done

# Find a free slot, blocking (polling) until one is available.
# Echoes the slot index on stdout.
acquire_slot () {
    while true; do
        for ((i = 0; i < MAX_CONCURRENT; i++)); do
            pid="${SLOT_PID[$i]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$i"
                return
            fi
        done
        # All slots busy -- wait for any background job to exit, then
        # re-check which slot freed up.
        wait -n 2>/dev/null
    done
}

run_job () {
    # $1 = job name, $2 = command to run
    local name="$1"
    local cmd="$2"

    local slot
    slot=$(acquire_slot)
    local gpu=$slot   # slot index doubles as the GPU id (0..MAX_CONCURRENT-1)
    local port=$((6000 + job_idx))

    (
        echo "Starting $name on GPU $gpu (slot $slot)"
        ZE_AFFINITY_MASK=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
            WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
            bash -c "$cmd"
        echo "Finished $name"
    ) &
    SLOT_PID[$slot]=$!

    job_idx=$((job_idx + 1))
}

#######################################################################
# shared dimension sweep
#   N_values = range(64, 2**12, 64)
#   for logb in range(5, 9): bench_list(b=2**logb, m=2048, N=N_values, k=2048)
#   -> b in {32, 64, 128, 256}, m=2048, k=2048 fixed, n swept.
#   One job per b value.
#######################################################################
for log_b in 5 6 7 8; do
    B=$((2**log_b))
    CMD="
python bmm_flops.py -b $B -m 2048 --n_range 64 4032 64 -k 2048 \
    --output_file results/sharedDimSweep_b${B}.csv \
    --notes 'shared dimension sweep: b=${B}, m=2048, k=2048, n swept'
"
    run_job "sharedDimSweep_b${B}" "$CMD"
done

#######################################################################
# effect of b on throughput with SQUARE individual MMs
#   for log_b in range(7): b = 2**log_b  (b in {1,2,4,8,16,32,64})
#       benchmark_bmm(b, 1024,1024,1024)
#       benchmark_bmm(b, 2048,2048,2048)
#       benchmark_bmm(b, 4096,4096,4096)
#       benchmark_bmm(b, 8192,8192,8192)
#   -b takes the full power-of-two list directly (independent dim);
#   one job per square size.
#######################################################################
for SZ in 1024 2048 4096 8192; do
    CMD="
python bmm_flops.py -b 1 2 4 8 16 32 64 -m $SZ -n $SZ -k $SZ \
    --output_file results/squareEffectOfB_${SZ}.csv \
    --notes 'square MM, b swept over powers of two: m=n=k=${SZ}'
"
    run_job "squareEffectOfB_${SZ}" "$CMD"
done

#######################################################################
# effect of b AND outer_dim on throughput with NON-SQUARE individual MMs
#   for log_b in range(7): b = 2**log_b
#       for log_outer_dim in range(5, 14): outer_dim = 2**log_outer_dim
#           benchmark_bmm_min(b, m=outer_dim, n=4096, k=outer_dim)
#   m=k=outer_dim paired (not independent of each other), n=4096 fixed,
#   b is independent -> pass the full -b list each call, loop over
#   outer_dim values (32..8192) within a single job.
#######################################################################
CMD='
for log_outer_dim in $(seq 5 13); do
    outer_dim=$((2**log_outer_dim))
    python bmm_flops.py -b 1 2 4 8 16 32 64 -m "$outer_dim" -n 4096 -k "$outer_dim" \
        --output_file results/nonSquareEffectOfB.csv \
        --notes "non-square MM: b swept 1..64, m=k=outer_dim=$outer_dim, n=4096"
done
'
run_job "nonSquareEffectOfB" "$CMD"

#######################################################################
# manual bmm example (h=2048, m=2048, k=h, n=h, b=512)
#   A:(b,m,n) B:(b,n,k) C:(b,m,k) = torch.bmm(A,B)
#######################################################################
CMD='
python bmm_flops.py -b 512 -m 2048 -n 2048 -k 2048 \
    --output_file results/manualBmmExample.csv \
    --notes "manual bmm example: h=2048, m=2048, n=2048, k=2048, b=512"
'
run_job "manualBmmExample" "$CMD"

wait
echo "All BMM sweeps complete."
