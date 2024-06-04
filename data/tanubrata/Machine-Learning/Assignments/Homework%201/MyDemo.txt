#!/bin/bash
#
#SBATCH --partition=gpu8_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=5-13:00:00
#SBATCH --mem=70GB
#SBATCH --mail-type=END
#SBATCH --mail-user=foo@bar.com 
#SBATCH --output=foo.txt

module load matlab/

module load anaconda3/cpu/5.2.0
module load cuda90/toolkit/9.1.176
module load cuda90/fft/9.1.176

cd /scratch/td2201/

python ml_is_good.py


