#!/usr/bin/env bash
#
#SBATCH --job-name dreambooth-inference
#SBATCH --output=dreambooth_inference_log.txt
#SBATCH --ntasks=1
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:1

#debug info
hostname
which python3
nvidia-smi

env

#activate env
source ~/.bashrc
cd /home/stud/xiec/anaconda3/bin
source activate
conda activate ldm

cd /home/stud/xiec/workarea/Dreambooth-Stable-Diffusion
python3 scripts/stable_txt2img.py --ddim_eta 0.0 \
	--n_samples 1 \
	--n_iter 4 \
	--scale 10.0 \
	--ddim_steps 80 \
	--ckpt /home/stud/xiec/workarea/Dreambooth-Stable-Diffusion/logdir/cat2022-12-18T10-38-11_dreambooth-training/checkpoints/last.ckpt \
	--prompt "a cross of a sks cat and a panda"
