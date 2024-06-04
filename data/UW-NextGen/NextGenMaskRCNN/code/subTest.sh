#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p slurm_courtesy

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:p100:1
#SBATCH -t 3-2:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

## Load CUDA into your environment
module load cuda/9.0


# (1)
# you need to create maskrcnn virtual environment first
source activate maskrcnn
# (2)
# normal environment setup
# install numpy
conda install --name maskrcnn numpy
conda install -c anaconda --name maskrcnn scikit-image
#conda install -c anaconda --name maskrcnn tensorflow-gpu==1.8
#
# Install Mask R-CNN using the Github Method
/srv/home/whao/anaconda3/envs/maskrcnn/bin/pip install -r requirements.txt
/srv/home/whao/anaconda3/envs/maskrcnn/bin/python3 setup.py install

## install cudatoolkit and cudnn
#conda install -c anaconda cudatoolkit==9.0 --yes
#conda install -c anaconda cudnn >=6.0.21 --yes
#
## install tensorflow and other libraries for machine learning
#pip install tensorflow-gpu==1.8
## install keras, skimage, mrcnn here:
#pip install keras==2.1.6   #2.1.5/6 DOESN'T WORK
#pip install scikit-image
#pip install mrcnn
#pip install numpy scipy scikit-learn pandas matplotlib seaborn
#pip install h5py
#pip install imgaug
#pip install scipy
#pip install Pillow
#pip install cython
#pip install matplotlib
#pip install opencv-python
#pip install imgaug
#pip install IPython
#
### Run Training Code

/srv/home/whao/anaconda3/envs/maskrcnn/bin/python3 ./samples/balloon/balloon.py train --dataset=./datasets/balloon --weights=coco





