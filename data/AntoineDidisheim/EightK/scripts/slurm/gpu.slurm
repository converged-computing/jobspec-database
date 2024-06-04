#!/bin/bash -l
#SBATCH --job-name="gpu_single"
#SBATCH --output=/home/adidishe/EightK/out/gpu.out
#SBATCH --error=/home/adidishe/EightK/out/gpu.err
#SBATCH --chdir=/home/adidishe/EightK

# Partition for the job:
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account="punim2039"
#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-0:10:00

module load foss/2022a
module load GCC/11.3.0
module load CUDA/11.7.0
module load GCCcore/11.3.0; module load Python/3.10.4

source /home/adidishe/EightK/venv/bin/activate
TensorFlow/2.11.0-CUDA-11.7.0

python3 gpu_explore.py