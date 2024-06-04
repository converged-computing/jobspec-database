#!/bin/bash -x
#SBATCH --account=slai
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=1:30:00
#SBATCH --partition=develbooster --gres=gpu:4

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SRUN_CPUS_PER_TASK}

ml Stages/2023 GCC OpenMPI CUDA imkl CMake Boost git

DMAX=10
N=30000

for i in {1..4}
do
srun --threads-per-core=1 ../ChASE/build/examples/2_input_output/2_input_output_mgpu --n $N --path_in=0 --isMatGen=true --dmax=${DMAX} --nev 2250 --nex 750 --complex 0 --tol 1e-10 --opt S --deg 20 --mode R --maxIter 1
done

