#!/bin/bash            
#SBATCH -A Project-ID        # your project_ID       
#SBATCH -J job-serial        # name of the job         
#SBATCH -n 1                 # nr. tasks  
#SBATCH --time=00:20:00      # requested time
#SBATCH --error=job.%J.err   # error file
#SBATCH --output=job.%J.out  # output file  
#SBATCH --gres=gpu:k80:1     # 1 GPU K80 card

ml purge  > /dev/null 2>&1
ml Julia/1.8.5-linux-x86_64
ml CUDA/11.4.1

julia <fix-activate-environment> <fix-name-script>.jl 
