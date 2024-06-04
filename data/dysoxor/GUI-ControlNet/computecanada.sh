#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --time=0-24:00
#SBATCH --account=def-gabilode
module load python/3.8 # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/control
source $SLURM_TMPDIR/control/bin/activate
#source /home/ansoba/projects/def-gabilode/ansoba/control/bin/activate
pip3 install numpy==1.24.2 --no-index
pip3 install torchvision==0.13.1 --no-index
pip3 install torch==1.12.1 --no-index
pip3 install -r /home/ansoba/projects/def-gabilode/ansoba/GUI-ControlNet/requirements.txt
cd $SLURM_TMPDIR
tar -xvzf /home/ansoba/projects/def-gabilode/ansoba/GUI-ControlNet/training/clay/data.tar.gz
export NCCL_BLOCKING_WAIT=1 # Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"
echo "$SLURM_NTASKS"
# tensorboard --logdir=/home/xiyex/scratch/xiyex/VPTR_ckpts/tensorboard --host 0.0.0.0 --load_fast false &
python /home/ansoba/projects/def-gabilode/ansoba/GUI-ControlNet/tool_add_control_sd21.py /home/ansoba/projects/def-gabilode/ansoba/GUI-ControlNet/models/v2-1_512-ema-pruned.ckpt /home/ansoba/projects/def-gabilode/ansoba/GUI-ControlNet/models/control_sd21_ini.ckpt
python /home/ansoba/projects/def-gabilode/ansoba/GUI-ControlNet/tutorial_train_sd21.py
deactivate