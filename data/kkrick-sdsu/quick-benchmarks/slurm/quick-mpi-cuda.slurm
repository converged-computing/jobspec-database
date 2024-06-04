#!/usr/bin/env bash

#SBATCH --job-name="quick-mpi-cuda"
#SBATCH --output="quick-out-mpi-cuda.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=93G
#SBATCH --gpus=4
#SBATCH --account=sdu135
#SBATCH --no-requeue
#SBATCH -t 01:00:00

module reset
module load singularitypro

singularity exec --nv --bind /expanse,/scratch \
    containers/quick_mpi-cuda-12.0.1.sif \
    ./bench.sh \
    -i ./input \
    -o ./output \
    -f psb5,morphine,taxol,valinomycin \
    -c quick.cuda.MPI
