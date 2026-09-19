#! /bin/bash -x
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=12:0:00
#PBS -q capacity
#PBS -A foundmedicine
#PBS -l filesystems=flare
#PBS -k doe
#PBS -N mm_sweep
#PBS -j oe

set +x
source /lus/flare/projects/FoundMedicine/jahatef/setup.sh
cd /lus/flare/projects/FoundMedicine/jahatef/sizing/cookbook/benchmarks/sizing

python batch_vs_concat.py

#!/bin/bash
# Rewrites the original benchmark_mm / benchmark_mm_b / benchmark_mm_concat
# experiment list using mm_flops.py.
#
# NOTE on batching: mm_flops.py has no -b/--batch flag. A batched GEMM
# benchmark_mm_b(m, n, k, b=B) does the same total FLOPs as a single GEMM
# of shape (B*m, n, k) (this is also what "concat" does), so every
# benchmark_mm_b(...) call below is rewritten as mm_flops.py with
# m replaced by (m * B). This loses the "B separate kernel launches vs.
# 1 launch" distinction from the original batch-vs-concat experiment,
# which can't be expressed through this CLI -- see EXP_BATCH_VS_CONCAT.
#
# NOTE on paired sweeps: --m_range/--n_range/--k_range each take an
# independent arithmetic range and (per the documented API) combinations
# across -m/-n/-k are NOT necessarily zipped 1:1, they're treated as
# independent dimensions. Several original experiments sweep two dims
# together with a fixed *relationship* (m=k, n=4k, n=3k, etc.) rather than
# independently. Those are expressed below as a bash loop issuing one
# mm_flops.py call per value, all appending to the same --output_file.

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
# Figure 3. basicGemmMKSweep -- m=k swept over powers of two, n=4096
#   for log_size in range(5, 14): benchmark_mm(2**log_size, 4096, 2**log_size)
#######################################################################
CMD='
for log_size in $(seq 5 13); do
    sz=$((2**log_size))
    python mm_flops.py -m "$sz" -n 4096 -k "$sz" \
        --output_file results/basicGemmMKSweep.csv \
        --notes "Figure3 basicGemmMKSweep: m=k=2^log_size, n=4096"
done
'
run_job "basicGemmMKSweep" "$CMD"

#######################################################################
# Figure 7. basicGemmKSweep -- k in [64, 32768), m=2048, n=2048
#   for k in range(64, 2**15, 64): benchmark_mm(2048, 2048, k)
#######################################################################
CMD='
python mm_flops.py -m 2048 -n 2048 --k_range 64 32704 64 \
    --output_file results/basicGemmKSweep.csv \
    --notes "Figure7 basicGemmKSweep: m=2048, n=2048, k swept"
'
run_job "basicGemmKSweep" "$CMD"

CMD='
python mm_flops.py -m 2048 -n 2048 --k_range 32704 524288 1 \
    --output_file results/basicGemmKSweep.csv \
    --notes "Figure7 basicGemmKSweep: m=2048, n=2048, k swept"
'

#######################################################################
# Figure 8. basicGemmLargeKSweep -- k in [1536, 6208), m=2304, n=4096
#   for k in range(1536, 6208, 64): benchmark_mm(2304, 4096, k)
#######################################################################
CMD='
python mm_flops.py -m 2304 -n 4096 --k_range 1536 6144 64 \
    --output_file results/basicGemmLargeKSweep.csv \
    --notes "Figure8 basicGemmLargeKSweep: m=2304, n=4096, k swept"
'
run_job "basicGemmLargeKSweep" "$CMD"

#######################################################################
# m sweep -- m in [64, 32768), n=2048, k=2048
#   for m in range(64, 2**15, 64): benchmark_mm(m, 2048, 2048)
#######################################################################
CMD='
python mm_flops.py --m_range 64 32704 64 -n 2048 -k 2048 \
    --output_file results/mSweep.csv \
    --notes "m sweep: n=2048, k=2048"
'
run_job "mSweep" "$CMD"

#######################################################################
# n sweep -- n in [64, 32768), m=2048, k=2048
#   for n in range(64, 2**15, 64): benchmark_mm(2048, n, 2048)
#######################################################################
CMD='
python mm_flops.py -m 2048 --n_range 64 32704 64 -k 2048 \
    --output_file results/nSweep.csv \
    --notes "n sweep: m=2048, k=2048"
'
run_job "nSweep" "$CMD"

#######################################################################
# nk sweep -- for nk in range(64, 2**15, 64): benchmark_mm(2048, 4*nk, nk)
#   m=2048 fixed, n=4*k, k swept -- paired, needs per-value loop
#######################################################################
CMD='
for k in $(seq 64 64 32704); do
    n=$((4 * k))
    python mm_flops.py -m 2048 -n "$n" -k "$k" \
        --output_file results/4nkSweep.csv \
        --notes "nk sweep: m=2048, n=4*k, k=$k"
done
'
run_job "4nkSweep" "$CMD"

#######################################################################
# mn sweep -- for mn in range(64, 4096, 8): benchmark_mm(mn, 2048, mn)
#   m=k=mn, n=2048 fixed -- paired, needs per-value loop
#######################################################################
CMD='
for mk in $(seq 64 8 4088); do
    python mm_flops.py -m "$mk" -n 2048 -k "$mk" \
        --output_file results/mkSweep.csv \
        --notes "mn sweep: m=k=$mk, n=2048"
done
'
run_job "mkSweep" "$CMD"

#######################################################################
# batch vs concat
#   for n in range(64, 4096, 64):
#       benchmark_mm_b(2048, n, 2048, b=4)        -> m_eff = 2048*4 = 8192
#       benchmark_mm_concat(2048, n, 2048, b=4)    -> m_eff = 2048*4 = 8192
#   Both reduce to the same FLOPs-equivalent GEMM shape on this CLI
#   (m=8192, n swept, k=2048); kept as two identically-shaped jobs so the
#   resulting CSVs preserve which original variant they came from.
#######################################################################
CMD='
python mm_flops.py -m 8192 --n_range 64 4032 64 -k 2048 \
    --output_file results/batchVsConcat_batched.csv \
    --notes "batch variant: orig m=2048,k=2048,b=4 -> m_eff=8192, n swept"
'
run_job "batchVsConcat_batched" "$CMD"

CMD='
python mm_flops.py -m 8192 --n_range 64 4032 64 -k 2048 \
    --output_file results/batchVsConcat_concat.csv \
    --notes "concat variant: orig m=2048,k=2048,b=4 concatenated -> m_eff=8192, n swept"
'
run_job "batchVsConcat_concat" "$CMD"

#######################################################################
# profile linear projection
#   benchmark_mm_b(4, 13056, 13056, b=2048) -> m_eff = 4*2048 = 8192
#######################################################################
CMD='
python mm_flops.py -m 8192 -n 13056 -k 13056 \
    --output_file results/linearProjection.csv \
    --notes "linear projection profile: orig m=4,n=13056,k=13056,b=2048 -> m_eff=8192"
'
run_job "linearProjection" "$CMD"

#######################################################################
# sweep n for B in {16, 32}
#   for logB in range(4,6): B=2**logB
#       for n in range(64, 2**15, 64): benchmark_mm_b(2048, n, 2048, b=B)
#   m_eff = 2048*B, n swept, k=2048
#######################################################################
for logB in 4 5; do
    B=$((2**logB))
    m_eff=$((2048 * B))
    GPU=$((job_idx % 6))
    CMD="
python mm_flops.py -m $m_eff --n_range 64 32704 64 -k 2048 \
    --output_file results/sweepN_B${B}.csv \
    --notes 'sweep n: orig m=2048,k=2048,b=${B} -> m_eff=${m_eff}, n swept'
"
    run_job "sweepN_B${B}" "$CMD"
done

#######################################################################
# sweep n in area of low speed
#   for hidden_size in range(22976, 25024+64, 64):
#       benchmark_mm_b(4, hidden_size, hidden_size, b=2048)
#   m_eff = 4*2048 = 8192, n=k=hidden_size paired -- per-value loop
#######################################################################
CMD='
for hs in $(seq 22976 64 25024); do
    python mm_flops.py -m 8192 -n "$hs" -k "$hs" \
        --output_file results/sweepNK_lowSpeedRegion.csv \
        --notes "low-speed-region sweep: orig m=4,b=2048 -> m_eff=8192, n=k=$hs"
done
'
run_job "sweepNK_lowSpeedRegion" "$CMD"

#######################################################################
# profile separate arbitrary region
#   for hidden_size in range(64, 2**15, 64):
#       benchmark_mm_b(4, 3*hidden_size, hidden_size, b=2048)
#   m_eff = 4*2048 = 8192, n=3*k, k swept -- paired, per-value loop
#######################################################################
CMD='
for hs in $(seq 64 64 32704); do
    n=$((3 * hs))
    python mm_flops.py -m 8192 -n "$n" -k "$hs" \
        --output_file results/arbitraryRegion.csv \
        --notes "arbitrary region sweep: orig m=4,b=2048 -> m_eff=8192, n=3*hs, k=hs=$hs"
done
'
run_job "arbitraryRegion" "$CMD"

#######################################################################
# h to 4h drop (the three active loops at the bottom of the original file)
#   loop1: benchmark_mm_b(2048, h, 3*h, b=4)        -> m_eff = 2048*4 = 8192
#   loop2: benchmark_mm_b(4, h, 3*h, b=2048)        -> m_eff = 4*2048 = 8192
#   loop3: benchmark_mm_b(4*2048, h, 3*h)           -> m_eff = 8192 (b=1)
#   All three collapse to the identical FLOPs-equivalent shape
#   (m=8192, n=h, k=3h) under this CLI since launch-count/batching can't
#   be distinguished -- kept as three separate jobs/output files so the
#   provenance of each is preserved even though the numbers will match.
#######################################################################
for variant in hTo4hDrop_loop1_b4 hTo4hDrop_loop2_b2048 hTo4hDrop_loop3_nob; do
    GPU=$((job_idx % 6))
    CMD="
for h in \$(seq 128 128 32768); do
    n=\$((3 * h))
    python mm_flops.py -m 8192 -n \"\$h\" -k \"\$n\" \
        --output_file results/${variant}.csv \
        --notes 'h to 4h drop (${variant}): m_eff=8192, n=h, k=3h'
done
"
    run_job "$variant" "$CMD"
done

wait
echo "All sweeps complete."
