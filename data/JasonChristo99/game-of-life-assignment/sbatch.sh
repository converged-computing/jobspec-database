#!/bin/bash
#SBATCH --job-name=game_of_life
#SBATCH --output=gol_%j.out
#SBATCH --error=gol_%j.err
#SBATCH --partition=thin_course
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=01:00:00  # minutes time limit

module purge
module load 2023
module load MPICH/4.1.2-GCC-12.3.0

# Array of total processes/threads
total_values=(1 2 4 8 16 32 64 128)


# Loop over total processes/threads
for total in "${total_values[@]}"; do
    echo "Testing for $total processes/threads"
    
    # Loop over possible combinations
    for p in $(seq 1 $((total))); do
        t=$((total / p))
        
        # Ensure the product of processes and threads equals the total
        if [ $((p * t)) -eq $total ]; then
            echo "Running with $p processes / $t threads per process"
            export OMP_NUM_THREADS=$t
            time mpirun -np $p --bind-to none ./game_of_life -omp $t
            echo "-----------DONE--------------"
        fi
    done
done
