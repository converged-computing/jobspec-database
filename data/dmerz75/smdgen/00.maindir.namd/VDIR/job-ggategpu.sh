#!/bin/bash
#PBS -N xxjobnamexx
#PBS -j oe
#PBS -l xxwalltimexx
#PBS -l pmem=220mb
#PBS -l xxnodesxx
#PBS -V

# job properties
NAMD_DIR=/opt/NAMD29_MCUDA/
export PATH=${NAMD_DIR}:${PATH}
export LD_LIBRARY_PATH=${NAMD_DIR}:${LD_LIBRARY_PATH}

# Parse out the GPU device number from the $PBS_GPUFILE
device=`grep -- "-gpu" $PBS_GPUFILE | sed 's/.*-gpu//'`

cd $PBS_O_WORKDIR

# job running
./go.py $device
