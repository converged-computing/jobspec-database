#!/bin/bash
#SBATCH --chdir /home/xogay
#SBATCH --partition=gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --time 00:05:00
#SBATCH --requeue

echo STARTING AT $(date)
echo "Job run at: $(hostname)"

module load gcc/8.4.0-cuda py-torchvision/0.6.1 python/3.7.7 mvapich2/2.3.4 py-tensorflow/2.3.1-cuda-mpi

source ~/.venvs/guided-diffusion-repaint/bin/activate
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate base
echo "Activated virtual environment:"
echo $CONDA_DEFAULT_ENV
echo "bash ${@:1}"
bash "${@:1}"

cd /home/xogay/Guided-Diffusion
pip install -e . --user
pip install mpi4py --user
python scripts/image_train.py --data_dir /home/xogay/Guided-Diffusion/datasets/gt --image_size 128 --num_channels 256 --num_res_blocks 2 --learn_sigma True --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 4 --save_interval 100
echo FINISHED AT $(date)