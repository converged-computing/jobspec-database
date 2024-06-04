#!/bin/bash
#SBATCH -N 1                     # number of nodes
#SBATCH --ntasks-per-node=48	  # number of cores per node
#SBATCH --error=job.%J.err	 # name of output file
#SBATCH --output=job.%J.out	 # name of error file
#SBATCH --time=24:00:00          # time required to execute the program
#SBATCH --partition=standard     # specifies queue name (standard is the defaul$
#module load DL-Conda_3.7
source activate /home/apps/DL/DL-CondaPy3.7

# Run Python script
python cmb.py
