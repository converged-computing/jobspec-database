#!/bin/bash

#A typical run takes couple of hours but may be much longer
#SBATCH --job-name=feature_compute
#SBATCH --time=10:00:00

#log files:
#SBATCH -e "/n/home10/ytingliu/alphapulldown_new/logs/%A_%a_err.txt"
#SBATCH -o "/n/home10/ytingliu/alphapulldown_new/logs/%A_%a_out.txt"

#qos sets priority
#SBATCH --qos=high
#SBATCH --ntasks=8

#SBATCH -p gpu_requeue
#SBATCH --cpus-per-task 8

#Reserve the entire GPU so no-one else slows you down
#SBATCH --gres=gpu:1

#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH --mem=128G

#SBATCH --mail-type=END
#SBATCH --mail-user=yutingliu@hsph.harvard.edu

module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01
module load python/3.10.9-fasrc01
conda activate alphapulldown_new

# In the modified sandbox,  the cutoff variable is already inactivated. Therefore just give it an random integer.
cutoff=50
bind_path=$1

singularity exec \
    --no-home \
    --bind "$bind_path":/mnt \
    /n/holyscratch01/ramanathan_lab/yuting/alpha_analysis \
    run_get_good_pae.sh \
    --output_dir=/mnt \
    --cutoff=$cutoff
