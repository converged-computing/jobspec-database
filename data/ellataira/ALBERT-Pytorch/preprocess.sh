#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=prepalbert412
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --output=logs/prepalbert412.%j.out
#SBATCH --error=logs/prepalbert412.%j.err

module load anaconda3/2022.05 cuda/12.1
#conda create --name pytorch_env python=3.9 -y
conda activate greenai
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -c'import torch; print(torch.cuda.is_available())'

cd /home/taira.e/ALBERT-Pytorch

python preprocess.py &

prep_id=$!

# Start continuous monitoring while the bert-vocab command is running
while ps -p $prep_id > /dev/null; do
   # Get timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Get GPU power draw and append to CSV file
    power_draw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    echo "$timestamp,$power_draw" >> /home/taira.e/power_stats/albertprep.csv

    sleep 300  # 5 mins

done

wait $prep_id

conda deactivate

