#!/bin/bash
#SBATCH -J GAN_test
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yw132@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node v100:1
#SBATCH --time=23:00:00


module load python/gpu
# pip install -r requirements.txt
pip install visdom
pip install dominate
python train.py --dataroot /N/project/polycrystalGAN/polycrystal/pytorch-CycleGAN-and-pix2pix/datasets/1000 --name 001 --model pix2pix --direction AtoB
python test_loop.py --dataroot /N/project/polycrystalGAN/polycrystal/pytorch-CycleGAN-and-pix2pix/datasets/1000 --direction AtoB --model pix2pix --name 001