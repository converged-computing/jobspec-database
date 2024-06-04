#!/bin/bash
#PBS -l select=128:ncpus=32:mpiprocs=8
#PBS -A ERDCV00898R40
#PBS -l walltime=024:00:00
#PBS -q standard_sm 
#PBS -N profileProteus
#PBS -j oe
#PBS -l application=proteus
#PBS -m be
#PBS -M aron@ahmadia.net
source /opt/modules/default/init/bash
module swap PrgEnv-pgi PrgEnv-gnu
module unload xt-libsci
module load acml
source /u/aron/proteus/garnet.gnu/bin/proteus_env.sh
PARUN=/u/aron/proteus/garnet.gnu/bin/parun
PROTEUS_PYTHON=$(which python)

export MPICH_RANK_REORDER_METHOD=2

procs_per_node=8
for i in 1 2 4 8 16 32 64 128 256 512 1024
do 
workdir=$PBS_O_WORKDIR/dambreak_scale_2D_$i
cp -R /u/aron/proteus-profile/scale/dambreak2D $workdir
cd $workdir
aprun -n $i ${PROTEUS_PYTHON} ${PARUN} dambreak_so.py -l 5 -v -O ./petsc.options.schur_upper_a11_asm_boomeramg 
done

