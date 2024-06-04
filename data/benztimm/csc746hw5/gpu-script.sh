#!/bin/bash 
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH -A m3930
#SBATCH -J queue
#SBATCH --job-name=gpu-job
#SBATCH --output=gpu-job.o%j
#SBATCH --error=gpu-job.e%j

# Load any modules or source your profile if needed to set up the environment

# Array of thread block sizes
threads_per_block=("32" "64" "128" "256" "512" "1024")

# Array of numbers of thread blocks
num_thread_blocks=("1" "4" "16" "64" "256" "1024" "4096")

# Loop over each size of thread block
for tpb in "${threads_per_block[@]}"
do
    # Loop over each number of thread blocks
    for ntb in "${num_thread_blocks[@]}"
    do
        echo "Running with ${tpb} threads per block and ${ntb} thread blocks"
        # Run the application with ncu for profiling
        ncu --set default --section SourceCounters \
            --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg \
            sobel_gpu $tpb $ntb
    done
done
