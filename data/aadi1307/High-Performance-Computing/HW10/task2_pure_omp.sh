#!/usr/bin/env zsh
#SBATCH --job-name=task2_pure
#SBATCH --partition=instruction
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task2pure.out
#SBATCH --mem=20G

module load nvidia/cuda/11.8.0


# Compile the task2 program
g++ task2 pure omp.cpp reduce.cpp -Wall -O3 -o task2 pure omp -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

# Fixed array size n
n=1000000

# Number of threads to test
t_values=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# Loop over the thread counts
for t in "${t_values[@]}"; do
    # Update the number of CPUs per task to match the number of threads
    # Note: This line is a placeholder and will not actually change the SBATCH directive.
    # You may need to adjust the SBATCH --cpus-per-task= directive manually or through a parameter substitution mechanism provided by your HPC system.
    #SBATCH --cpus-per-task=$t

        ./task2 $n $t
done


