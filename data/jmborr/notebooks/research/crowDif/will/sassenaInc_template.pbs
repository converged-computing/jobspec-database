#!/bin/bash

#PBS -N _SYSTEM_
#PBS -j oe
#PBS -l walltime=06:00:00
#PBS -l mppwidth=24
host=hopper

# Load machine-dependent lammps
source $MODULESHOME/init/bash
if [[ $host = hopper* ]]; then
  module load sassena
fi

cd $PBS_O_WORKDIR
aprun -n 24 sassena --config=sassenaInc.xml &> sassenaInc.log
#wait  # only needed when submitting several executables

# Copy output and input to project directory
PROJDIR=/project/projectdirs/m1503/jmborr/crowDif/will
mkdir -p $PROJDIR
cd $PBS_O_WORKDIR
/bin/cp sassena*_SYSTEM_* fqt*_SYSTEM_* $PROJDIR/



####### EDISON Bath Queue Walltimes as a Function of Number of Nodes #######
#Note: 1 node = 24 cores
# Queue        Nodes Wallclock Priority RunLimit QueuedLimit ChargeFactor
# debug        1-512  30 mins      2        2         2           1
# reg_small    1-682  48 hrs       3       24        24           1
# reg_med    683-2048 36 hrs       3        8         8           0.6
# reg_big   2049-4096 36 hrs       2        2         2           0.6
# reg_xbig  4097-5462 12 hrs       2        2         2           0.6
# premium      1-2048 12 hrs       1        1         1           2
# low          1-682  24 hrs       4        8         8           0.5
################################

####### CARVER 
#Note: 1 node = 24 cores
# Queue        Nodes Wallclock Priority RunLimit QueuedLimit ChargeFactor
# debug        1-32   30 mins      2        2         1           1.0
# reg_short    1-16    4 hrs       3        8         4           1.0
# reg_small    1-16   48 hrs       3        6         3           1.0
# reg_med     17-32   36 hrs       3        5         2           1.0
# reg_big     33-64   24 hrs       3        3         1           1.0
# reg_long     1-16  168 hrs       3        2         1           1.0
# reg_xlong    1-4   504 hrs       3        2         1           1.0
# low          1-32   24 hrs       4        5         3           0.5
# serial       1      48 hrs       3      150        20           1.0
################################

# Some useful tidbits:
#  Qeue usage info at https://www.nersc.gov/users/queues/queue-wait-times/
#     qsub -W depend=afterok:123451 2.pbs  chain submissions
#     showbf   shows unallocated Nodes

# Setting number of cores in each cluster, running parallel jobs
#        Edison                Hopper                Carver
#  PBS -l mppwidth=24     PBS -l mppwidth=24    PBS -l nodes=3:ppn=8
#     aprun -np 24          aprun -np 24           mpirun -np 24
