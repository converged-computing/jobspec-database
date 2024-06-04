#!/bin/bash
#SBATCH --job-name=orthoRes
#SBATCH --output="log/othroRes.out"
#SBATCH --error="error/orthoRes.err"
#SBATCH --partition=gpux2
#SBATCH --nodes=1
#SBATCH --time=24
#SBATCH --cpu_per_gpu=40
#SBATCH --mem-per-cpu=2048
#SBATCH --begin=now+0minutes

echo Running
module load opence/1.5.1
echo Module loaded

python runme_four.py -e 100 -t ../data/
echo Done
