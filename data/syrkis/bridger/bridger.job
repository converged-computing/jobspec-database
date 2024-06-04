#!/bin/bash

#SBATCH --job-name=test            # Job name
#SBATCH --output=outs/%x.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=1        # Schedule one core
#SBATCH --time=01:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --gres=gpu
#SBATCH --partition=brown    # Run on either the Red or Brown queue
# #SBATCH --mail-type=FAIL,END          # Send an email when the job finishes or fails
#SBATCH --mem=30000             # memo

module load singularity
module --ignore-cache load CUDA
singularity run --nv -B /home/common/datasets/amazon_review_data_2018,/home/timp/repositories/bringo/data bridger.sif
# ./scripts/counts.sh
# ./scripts/origin.sh

