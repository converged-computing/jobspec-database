#!/bin/bash
#PBS -l select=64:ncpus=32:mpiprocs=4
#PBS -A ERDCV00898R40
#PBS -l walltime=06:00:00
#PBS -q R755829 
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

procs_per_node=4
for i in 1 2 4 8 16 32 64 128 256 
do 
workdir=$PBS_O_WORKDIR/dambreak_scale_2D_short_trace_heavy_$i
cp -R /u/aron/proteus-profile/scale/dambreak2D_short_trace_force $workdir
cd $workdir
aprun -n $i ${PROTEUS_PYTHON} ${PARUN} dambreak_so.py -l 5 -v -O ./petsc.options.schur_upper_a11_asm_boomeramg 
done

