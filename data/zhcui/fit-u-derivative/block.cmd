#!/bin/bash

##SBATCH --qos=heavy
##SBATCH --qos=normal
#SBATCH --partition=parallel
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH -c 28
#SBATCH -t 100:00:00
#SBATCH --mem=250000

export SLURM_MPI_TYPE=pmi2
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

module load gcc-5.4.0/boost-1.55.0-openmpi-1.10.3 

source /home/zhcui/.bashrc

srun hostname
ulimit -l unlimited
python ./hub2d.ti.py 4.0 . 0.9

#export SCRATCHDIR="/scratch/local/zhcui/uray/hub4x4_M1000_28"
#srun mkdir -p $SCRATCHDIR
#srun rm -r $SCRATCHDIR
#srun mkdir -p $SCRATCHDIR
#srun /home/shengg/opt/stackblocklatest/stackblock/block.spin_adapted dmrg.conf > dmrg.out
#rm -r  $SCRATCHDIR/node0/Block*
#mv $SCRATCHDIR/node0 node_new
#srun rm -r $SCRATCHDIR
