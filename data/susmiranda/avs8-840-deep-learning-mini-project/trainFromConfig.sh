#!/bin/bash

# Source: https://stackoverflow.com/a/44168719
# Run using the following: bash trainFromConfig.sh <Config_Name>

sbatch <<EOT
#!/bin/bash
#SBATCH --output="train_${1}.log"
#SBATCH --job-name="train_${1}"
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00 # walltime

echo "The config file used is KWT_configs/${1}.cfg"
srun --gres=gpu:1 singularity exec --nv ~/pytorch-24.01 python train.py --conf "KWT_configs/${1}.cfg"
EOT
