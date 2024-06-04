#!/bin/bash
#SBATCH -J solar_05
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 32
#SBATCH --ntasks-per-node=8
#SBATCH -n 256
#SBATCH --account=m1759
##SBATCH --gres=gpu:4
##SBATCH -C tesla
##SBATCH --qos normal
#SBATCH -o hostname_%j.out
#SBATCH -e hostname_%j.err


module load pytorch/v1.0.1

#source activate pytorch_mpi
#export PYTHONPATH=/sdcc/u/kyu/.conda/envs/pytorch_mpi/lib/python3.5/site-packages/

#export PATH=/sdcc/u/kyu/lib/openmpi_cuda/bin:$PATH
#export LD_LIBRARY_PATH=/sdcc/u/kyu/lib/openmpi_cuda/lib:$LD_LIBRARY_PATH


which python

echo "==================================="
echo ""

srun python ../../LSTNet_MPI_cpu.py --hidSkip 10 --batch_size 16 --data_amp_size 1 --epochs 20 --gpu 8 --data ../../data/solar_AL.txt --save ../../save/solar.pt --output_fun Linear > screen_solar_001N_08n.out
