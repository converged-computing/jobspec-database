#!/bin/bash

#SBATCH --job-name=replication-gpu    # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8       # Schedule one core
#SBATCH --gres=gpu:v100:1              # Schedule a GPU
#SBATCH --time=03:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red,brown    # Run on either the Red or Brown queue

echo "Running on $(hostname):"
module load Anaconda3/
eval "$(conda shell.bash hook)"
conda activate reppsych
# pip install pynvml
# pip install packaging
# pip install tqdm
# pip install requests
# pip install yaml
# pip install colorama
# pip install lxml
# pip install bitarray
# pip install cffi
# pip install cython
# pip install -r requirements.txt
python go.py
