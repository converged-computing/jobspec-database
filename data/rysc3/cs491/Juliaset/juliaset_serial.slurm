#!/bin/bash
#SBATCH --job-name serial_julia
#SBATCH --partition general
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 5:00
#SBATCH --output=julia_serial_%a.out
#SBATCH --array 1-7
#SBATCH --mail-user mfricke@unm.edu

OUTPUT_PATH=juliaset_weak_$SLURM_ARRAY_TASK_ID.tga

SCALE_FACTOR=$((2 ** ($SLURM_ARRAY_TASK_ID-1)))

export OMP_NUM_THREADS=1
FRACTAL_HEIGHT=$((100*$SCALE_FACTOR))
FRACTAL_WIDTH=200

echo Scale Factor: $SCALE_FACTOR
echo Number of Threads: $OMP_NUM_THREADS
echo FRACTAL_DIMS: $FRACTAL_HEIGHT x $FRACTAL_WIDTH

./juliaset $FRACTAL_HEIGHT $FRACTAL_WIDTH $OUTPUT_PATH

