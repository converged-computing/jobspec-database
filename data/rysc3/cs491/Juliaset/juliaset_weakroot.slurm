#!/bin/bash
#SBATCH --job-name weakroot_julia
#SBATCH --partition general
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 5:00
#SBATCH --output=julia_weakroot_%a.out
#SBATCH --array 1-7
#SBATCH --mail-user mfricke@unm.edu

OUTPUT_PATH=juliaset_weak_$SLURM_ARRAY_TASK_ID.tga

SCALE_FACTOR=$((2 ** ($SLURM_ARRAY_TASK_ID-1)))

export OMP_NUM_THREADS=$SCALE_FACTOR
FRACTAL_HEIGHT=$(bc <<< "scale=0; 100*sqrt($SCALE_FACTOR)")
FRACTAL_WIDTH=200

echo Scale Factor: $SCALE_FACTOR
echo Number of Threads: $OMP_NUM_THREADS
echo FRACTAL_DIMS: $FRACTAL_HEIGHT x $FRACTAL_WIDTH

./juliaset $FRACTAL_HEIGHT $FRACTAL_WIDTH $OUTPUT_PATH

