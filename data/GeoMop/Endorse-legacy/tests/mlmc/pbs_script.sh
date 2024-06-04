#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=1:ncpus=16:cgroups=cpuacct:mem=16Gb
#PBS -l walltime=2:00:00
#PBS -q charon_2h
#PBS -N endorse
#PBS -j oe

set -x

which python3

# run from pbs script directory
cd $PBS_O_WORKDIR || exit
pwd

#rm -rf /storage/liberec3-tul/home/martin_spetlik/MLMC_metamodels/comparison/data/mesh_size/cl_0_1_s_1/L1_4_T9_CNN
# ls -l

# collect arguments:
# singularity_exec script path
SING_SCRIPT="/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc/singularity_exec.py"
# singularity SIF image path (preferably create in advance)
#SING_FLOW="/storage/liberec3-tul/home/martin_spetlik/mlmc_metamodel_latest.sif"
SING_FLOW="/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc/endorse.sif"
# SING_FLOW="$HOME/workspace/flow123d_images/flow123d_3.0.5_92f55e826.sif"

# container mpiexec path (if not defined, default 'mpiexec' is used)
#IMG_MPIEXEC="/usr/local/mpich_3.4.2/bin/mpiexec"
# program and its arguments
PROG="python3 fullscale_transport_pbs.py run ../ --clean"

# directory with input files, all will be copied to $SCRATCHDIR
# SCRATCH_COPY="$PBS_O_WORKDIR/input"
# file contains list of input files, all will be copied to $SCRATCHDIR

#SCRATCH_COPY="/storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/"

#DATA_DIR_PATH="/storage/liberec3-tul/home/martin_spetlik/MLMC_metamodels/comparison/data/mesh_size/cl_0_1_s_1/"
#DATA_DIR_1="L1_4_50k"
#DATA_DIR_2="L1_4"
#DIRS_TO_SCRATCH="${DATA_DIR_PATH}${DATA_DIR_1}.tar.xz ${DATA_DIR_PATH}${DATA_DIR_2}.tar.xz /storage/liberec3-tul/home/martin_spetlik/MLMC_metamodels/comparison/data/mesh_size/l_step_1.0_common_files"

cd /storage/liberec3-tul/home/martin_spetlik/Endorse_full_transport
singularity exec $SING_FLOW ./setup.sh

#singularity exec --bind /usr/bin/qsub $SING_FLOW
#singularity exec --bind /usr/bin/qstat $SING_FLOW



cd /storage/liberec3-tul/home/martin_spetlik/Endorse_MS_full_transport/tests/mlmc

#singularity exec docker://flow123d/endorse:latest python3 fullscale_transport_pbs.py run ../ --clean
#singularity exec $SING_FLOW python3 -m pip install numpy==1.21.0
#
#singularity exec $SING_FLOW python3 -m pip install /storage/liberec3-tul/home/martin_spetlik/MLMC_metamodels/comparison/MLMC_meta_pytorch
#singularity exec $SING_FLOW python3 -m pip install tensorflow-datasets==4.6.0 gstools ruamel.yaml memoization seaborn spektral

#python3 $SING_SCRIPT -i $SING_FLOW -m None -s $SCRATCH_COPY -ds $DIRS_TO_SCRATCH -- "$PROG"
#python3 $SING_SCRIPT -i $SING_FLOW -m None -s $SCRATCH_COPY -- "$PROG"
python3 $SING_SCRIPT -B /usr/bin/qsub,/usr/bin/qstat -i $SING_FLOW -m None  -- "$PROG"

## possibly copy the results from scratch
#if [ ! -d "$PBS_O_WORKDIR/output" ]; then
#  mkdir $PBS_O_WORKDIR/output
#fi
#ls -l $SCRATCHDIR
##cp -r $SCRATCHDIR/.  $PBS_O_WORKDIR/output/
#rsync -a $SCRATCHDIR/. $PBS_O_WORKDIR/output/ --exclude $DATA_DIR_1 --exclude $DATA_DIR_2
#
#clean_scratch