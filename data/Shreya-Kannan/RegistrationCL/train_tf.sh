#!/bin/bash
#SBATCH --account=rrg-punithak
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --mem=32G
#SBATCH --time=35:00:00
#SBATCH --mail-user=skannan3@ualberta.ca
#SBATCH --mail-type=ALL

module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install voxelmorph
pip install tensorflow
pip install numpy==1.23.5
pip install "nibabel<5"

NOW=$(date '+%Y%m%d%H%M%S')
python /home/shreya/scratch/train_tf_nmi.py
#python /home/shreya/scratch/voxelmorph/scripts/tf/train.py --img-list '/home/shreya/scratch/voxelmorph/images/AbdomenMRCT/AbdomenMRCT_dataset.json'  --epochs 500 --image-loss "nmi" --lambda 1 --model-dir /home/shreya/scratch/voxelmorph/models/tf/$NOW
