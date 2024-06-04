#!/bin/bash
#SBATCH --job-name=prep_gromacs
#SBATCH --ntasks=1
#SBATCH --partition=hpc
#SBATCH --output=gromacs-prep-%j.out
#SBATCH --error=gromacs-prep-%j.err

export TUT_DIR=$HOME/udocker-tutorial
export PATH=$HOME/udocker-1.3.10/udocker:$PATH
cd $TUT_DIR
export UDOCKER_DIR=$TUT_DIR/.udocker
module load python

echo "###############################"
hostname
echo ">> udocker command"
which udocker
echo
echo ">> List images"
udocker images
echo
echo ">> Create container"
udocker create --name=grom_mpi gromacs-mpi
echo
echo ">> Set execmode to F3"
udocker setup --execmode=F3 grom_mpi
echo
echo ">> List containers"
udocker ps -m -p
