#!/bin/bash
#SBATCH --job-name=run_gromacs
#SBATCH --ntasks=64
#SBATCH --partition=hpc
#SBATCH --output=gromacs-%j.out
#SBATCH --error=gromacs-%j.err

export TUT_DIR=$HOME/udocker-tutorial
export PATH=$HOME/udocker-1.3.10/udocker:$PATH
export UDOCKER_DIR=$TUT_DIR/.udocker
export OUT_NAME=output/ud-tutorial
export TRR=${OUT_NAME}.trr
export XTC=${OUT_NAME}.xtc
export EDR=${OUT_NAME}.edr
export LOG=${OUT_NAME}.log
module load python
cd $TUT_DIR

echo "###############################"
udocker run -v=$TUT_DIR/gromacs:/home/user -w=/home/user grom \
    gmx mdrun -s /home/user/input/md.tpr -e $EDR -x $XTC -o $TRR -g $LOG \
    -maxh 0.50 -resethway -noconfout -nsteps 10000 -nt 8 -pin on
