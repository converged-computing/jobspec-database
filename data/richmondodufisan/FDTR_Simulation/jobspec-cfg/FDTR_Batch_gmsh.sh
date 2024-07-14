#!/bin/bash
#SBATCH --account=p32089
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=0:30:00

python3 FDTR_mesh.py >> output.txt
