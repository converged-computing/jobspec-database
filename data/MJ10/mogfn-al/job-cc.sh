#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gres=gpu:1                         # Ask for 1 GPU
#SBATCH --mem=8G                             # Ask for 10 GB of RAM
#SBATCH --time=6:00:00                        # The job will run for 3 hours
#SBATCH -o /scratch/mjain/logs/lambo-%j.out  # Write the log on tmp1

module load python/3.8 cuda/11.1
export PYTHONUNBUFFERED=1

module load python/3.8
cd $SLURM_TMPDIR/
virtualenv --no-download venv
source venv/bin/activate

cd ~/lambo
pip install --no-index --find-links=~/wheels/ -r requirements-cc.txt
pip install -e .
pkill -9 wandb
python scripts/black_box_opt.py "$@"