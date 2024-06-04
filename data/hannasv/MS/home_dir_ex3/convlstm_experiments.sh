#!/bin/bash
#SBATCH -p dgx2q                  # partition (queue)
#SBATCH -N 1                      # number of nodes
#SBATCH -n 16                     # number of cores
#       #SBATCH -w g001                   # for a specific nod
#SBATCH --gres=gpu:1              # for e.g. 1 V100 GPUs
#SBATCH --mem 1024G               # memory pool for all cores
#SBATCH -t 1-24:00                # time (D-HH:MM)
#SBATCH -o output/slurm.%N.%j.out # STDOUT
#SBATCH -e output/slurm.%N.%j.err # STDERR
#     #SBATCH --exclusive         # If you want to benchmark and have node for yourself
#SBATCH --mail-user=hannasv@fys.uio.no
#SBATCH --mail-type=ALL

ulimit -s 10240
mkdir -p ~/output

. py37-venv/bin/activate # activate project enviornment

module purge
module load slurm/18.08.9
module load mpfr/gcc/4.0.2
module load gmp/gcc/6.1.2
module load mpc/gcc/1.1.0
module load gcc/9.1.1
module load openmpi/gcc/64/1.10.7
#module load openmpi/gcc/64/4.0.2
module load ex3-modules
module load cuda10.1/toolkit/10.1.243

module load python/3.7.4
#module load python3.7/scipy/1.3.1
#module load python3.7/xarray/0.15.1 

#srun -n $SLURM_NTASKS  /home/hannasv/a.out
# a.out er standard for kompilerte program i Unix/Linux ..
python /home/hannasv/MS/sclouds/ml/ConvLSTM/config_convlstm_models.py 
#srun -n $SLURM_NTASKS 1 python /home/hannasv/MS/sclouds/ml/ConvLSTM/config_model_2.py 
#srun -n $SLURM_NTASKS 1 python /home/hannasv/MS/sclouds/ml/ConvLSTM/config_model_3.py 
#srun -n $SLURM_NTASKS 1 python /home/hannasv/MS/sclouds/ml/ConvLSTM/config_model_4.py 
#srun -n $SLURM_NTASKS 1 python /home/hannasv/MS/sclouds/ml/ConvLSTM/config_model_5.py

