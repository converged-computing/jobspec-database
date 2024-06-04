#!/bin/bash
#SBATCH -J train-ecg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH -p dgx2q

echo "Loading modules"
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/20.02.7
module load pytorch-py37-cuda11.2-gcc8/1.9.1

if [ ! -f /usr/lib/x86_64-linux-gnu/libevent_core-2.1.so.6 ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/lib
fi

# 'VentRate', 'qt', 'pr', 'qrs', 'STJ_v5', 'T_PeakAmpl_v5', 'R_PeakAmpl_v5'
srun python train_medians.py -t qt -o stevennet_take5_x
#srun python train_medians.py -t pr -o stevennet_take5
#srun python train_medians.py -t qrs -o stevennet_take5
#srun python train_medians.py -t STJ_v5 -o stevennet_take5
#srun python train_medians.py -t T_PeakAmpl_v5 -o stevennet_take5
#srun python train_medians.py -t R_PeakAmpl_v5 -o stevennet_take5
#srun python train_medians.py -t VentRate -o stevennet_take5

