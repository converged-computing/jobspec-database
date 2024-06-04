#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A username
#SBATCH -J job-name
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --time=15-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@mail.com


	#for all queues
#module load centos7.3/app/namd/whole-runs-multicore


	#for cuda queue
module load centos7.3/app/namd/2017-11-10-multicore-cuda


echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"

$NAMD_DIR/namd2  +p $SLURM_NTASKS  +idlepoll config.conf
