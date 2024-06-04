#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --time=00:10:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1
#SBATCH --gpus=k80:1           # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --job-name openMP      # you can give your job a name for easier identification (same as -J)
#SBATCH --mem-per-cpu=100G            # memory required per allocated CPU (or core)

########## Command Lines for Job Running ##########

module load GCC/6.4.0-2.28 OpenMPI  ### load necessary modules.
module load cuDNN/8.4.1.50-CUDA-11.6.0
module load NCCL/2.8.3-CUDA-11.1.1 

cd /mnt/home/gaudisac/pGPT/        ### change to the directory where your code is located.

srun -n 1 ./hello.o     ### call your executable. (use srun instead of mpirun.)
scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file.
###js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powertools command).