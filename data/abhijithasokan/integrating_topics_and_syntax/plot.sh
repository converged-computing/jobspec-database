#!/bin/bash
#SBATCH -J ITS_plot
#SBATCH -t 3600
#SBATCH -n 12
#SBATCH -o ./plot.%j.out
#SBATCH -e ./plot.%j.err
#SBATCH --mail-type=END
#SBATCH --mem=1G

module purge
module load anaconda3/latest

. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate integrating_topics_syntax
python plot.py --alpha=0.02 --beta=0.02 --gamma=0.02 --delta=0.02 --num_iter=6000 --num_topics=10 --num_classes=8 --dataset=data2000 --test_dataset=train --test_dataset_size=2000
conda deactivate