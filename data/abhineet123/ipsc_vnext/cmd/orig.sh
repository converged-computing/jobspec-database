#!/bin/bash
#SBATCH --account=def-nilanjan
#SBATCH --nodes=1
#SBATCH --mem=500M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1


#SBATCH --job-name=imgn_r50_V4
#SBATCH --output=%x_%j.out

#SBATCH --time=0-00:30            # time (DD-HH:MM)

#SBATCH --mail-user=asingh1@ualberta.ca
#SBATCH --mail-type=ALL

module load python/3.6
# module load arch/avx512 StdEnv/2018.3 gcc cuda cudnn opencv/3.4.3

#TODO Change virtualenv source dir?
# source .envpy36/bin/activate
nvidia-smi

localdata=false

# Prepare data
if [ $localdata = true ]
then
    datasetdir=~/scratch/dataset/
else
    datasetdir=$SLURM_TMPDIR/imagenet/
    # move the imagenet data over to the slurm tmpdir
    # Prepare data, check dataset existence --> usefull with salloc and debugging
    if [ ! -d $datasetdir/train ]
    then
        mkdir -p $datasetdir/train
        cd $datasetdir/train
        echo 'Copying training data in ... '$PWD
        time tar xf ~/scratch/dataset/ILSVRC2012_img_train.tar -C .
        time find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xf "$PWD/${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    fi

    if [ ! -d $datasetdir/val ]
    then
        mkdir -p $datasetdir/val
        cd $datasetdir/val
        echo 'Copying validation data in ... '$PWD
        time tar xf ~/scratch/dataset/ILSVRC2012_img_val_modified.tar.gz -C .
    fi
fi

echo 'dataset directory: '$datasetdir

python main.py dataset=imgn budget=0.25 model=r50 ckpt_dir=log/imgn_r50_V4 workers=8 parallel=1 ep_train=30 ep_finetune=20 ep_trainslow=10 imagenet_path=$datasetdir

