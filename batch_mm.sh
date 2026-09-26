#! /bin/bash 
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:0:00
#PBS -q capacity
#PBS -A foundmedicine
#PBS -l filesystems=flare
#PBS -k doe
#PBS -N mm_prof_large
#PBS -j oe

source /lus/flare/projects/FoundMedicine/jahatef/setup.sh
cd /lus/flare/projects/FoundMedicine/jahatef/sizing/cookbook/benchmarks/sizing



module load xpu-smi

ZE_AFFINITY_MASK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=6000 \
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
python mm_flops.py -m 2048 -n 2048 --k_range 64 131072 1 \
    --output_file median/basicGemmKSweep.csv \
    --notes "Figure7 basicGemmKSweep: m=2048, n=2048, k swept"

for log_size in $(seq 5 17); do
    sz=$((2**log_size))
    python mm_flops.py -m "$sz" -n 4096 -k "$sz" \
        --output_file median/basicGemmMKSweep.csv \
        --notes "Figure3 basicGemmMKSweep: m=k=2^log_size, n=4096"
done
