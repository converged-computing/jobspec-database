#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --account def-bentahar
#SBATCH --cpus-per-task=4  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=125G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=5-00:00     # DD-HH:MM:SS
#SBATCH --array=1-$2
#SBATCH --output=/home/fgrcl/scratch/job-logs/slurm-%A_%a.out

module load python/3.10 cuda cudnn
source venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

export $(cat .env | xargs)
wandb agent --count 1 $1
EOT