#!/bin/bash

# Active comments for SLURM
#SBATCH -n 1                  # One task
#SBATCH -c 1                 # One cpu per task
#SBATCH -N 1                   # Minimum one node
#SBATCH -t 7-00:00:00            # Runtime in D-HH:MM
#SBATCH -p main # Partition to submit to
# #SBATCH --mem-per-cpu=4000   # Memory pool for all cores (see also --mem-per-cpu)

# Optional arguments (uncomment them to use)
#SBATCH --output=testslurm%N.%j.out    # Output file
#SBATCH --error=slurm_script.%N.%j.err     # Error output file
#SBATCH --mail-user=chamberlian1990@gmail.com  # User e-mail
#SBATCH --mail-type=FAIL         # When to send e-mail
#SBATCH --mem=32G # 16GB of memory
# #SBATCH --array=1-3

cd $PWD
# echo "hello slurm" > /home/lz210/dev/hello_test
# module purge

# module load python/3.5.2   intel/17.0.4
module load py3-numpy/1.14.3 py3-scipy/1.1.0
# srun python3 linearizedOrderingGenerator.py 4_2_2_2 $SLURM_ARRAY_TASK_ID
# srun python3 linearizedOrderingGenerator.py 4_2_2_2 1 
srun python3 samplerun.py 64 64prec3.dat  
