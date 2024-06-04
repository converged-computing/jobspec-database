#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=10gb:scratch_ssd=10gb:cluster=galdor
#PBS -N env_installation

# Load data and scripts
DATADIR=<path-to-your-directory-on-frontend-node-with-repo>
cp -r $DATADIR $SCRATCHDIR
cd $SCRATCHDIR/<name-of-your-directory-on-frontend-node-with-repo>

# Load and init conda module
export MODULEPATH=$MODULEPATH:<your-modules-directory-on-frontend-node>
module load <name-of-the-conda-module-file>
conda init
source ~/.bashrc

# Install conda environment
conda create --name islets-instance-segmentation python=3.8 -y
conda activate islets-instance-segmentation
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

conda install -c conda-forge progress -y
conda install -c conda-forge wandb -y

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

git clone --branch v3.2.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..

cp -r ./mmdetection $DATADIR

# Clean scratch
clean_scratch
