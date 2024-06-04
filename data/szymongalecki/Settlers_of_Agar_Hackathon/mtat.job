#!/bin/bash

#SBATCH --job-name=pytorch-gpu-condaenv    # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --time=23:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --mem=200G         
#SBATCH --gres=gpu
#SBATCH --partition=brown    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END
echo "Running on $(hostname):"

module load Anaconda3
conda create --name pytorchenv
source activate pytorchenv
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Verify install
python -c "import torch; print(torch.cuda.get_device_name(0))"
#pip3 install pandas==2.0.3
#pip3 install opencv-python
python3 model1.py