#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########
#SBATCH --time=30:00:00
#SBATCH -N 1 -n 2 -c 1
#SBATCH --mem-per-cpu=64            # memory required per allocated CPU (or core) - amount of memory (in bytes)        
#SBATCH --job-name=test
#SBATCH --mail-user=lixiaoy5@msu.edu
#SBATCH --mail-type=ALL


########## Command Lines to Run ##########
# module swap GNU Intel
# module purge
# module load GCC/9.3.0
# module load OpenMPI/4.0.3
# module load GCC/8.3.0 OpenMPI/3.1.4 R/4.0.2
# cd ${SLURM_SUBMIT_DIR}       ### change to the directory where your code is located

### call your executable (similar to mpirun)

MATLAB/2021a Pixel_js_analysis.m

scontrol show job $SLURM_JOB_ID           ### write job information to output file
js -j $SLURM_JOB_ID 