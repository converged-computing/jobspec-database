#!/bin/bash
#SBATCH --job-name=llama-cpp-map
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=map.out
#SBATCH --error=map.err

if [ "$USER" == "filippo.bistaffa" ]
then
    spack load --first py-pandas
else
    module load python/3.9.9
fi

srun python3 map.py --seed $RANDOM
