#!/bin/bash
###### Job name ######
#PBS -N tng_halo02
###### Output files ######
#PBS -o tng_halo02.out
#PBS -e tng.err
###### Number of nodes and cores ######
#PBS -l select=1:ncpus=64:mpiprocs=64:ompthreads=1
###### Specify how many hours do you need ######
#PBS -l walltime=24:00:00
###### Queue name ######
#PBS -q ct64
###### Sends mail to yourself when the job begins and ends ######
#PBS -M myho@asiaa.sinica.edu.tw
#PBS -m be
###### Sandbox ######
#PBS -W sandbox=PRIVATE
###### Specific the shell types ######
#PBS -S /bin/bash

###### Enter this job's working directory ######
cd $PBS_O_WORKDIR

###### Load modules to setup environment ######
. /etc/profile.d/modules.sh
module purge
module load intel/2022.0.2 mpi/2021.5.1
# module load gcc/11.2.0 openmpi/4.1.4 szip hdf5/1.12.1 python/3.10.2 fftw/2.1.5 gsl/2.7
module load szip hdf5/1.12.1 python/3.10.2 fftw/2.1.5 gsl/2.7

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/opt/fftw/2.1.5/mpi/2021.5.1/intel/2022.0.2/lib

###### Run your jobs with parameters ######
if [ -n "$PBS_NODEFILE" ]; then
  if [ -f $PBS_NODEFILE ]; then
    NPROCS=`wc -l < $PBS_NODEFILE`
  fi
fi

#$OPENMPI_HOME/bin/mpirun -v -machinefile $PBS_NODEFILE -np $NPROCS ./a.out foo=bar
# mpirun -np 1 ./GIZMO TNG/test/params.txt
# cp GIZMO_ics Config.sh output/TNG/Halo00/
# mpirun -machinefile $PBS_NODEFILE -np 1 ./GIZMO_ics TNG/Halo00/params_ics.txt
cp GIZMO_ics_2 output/TNG/Halo02/
mpirun -machinefile $PBS_NODEFILE -np $NPROCS ./GIZMO_ics_2 TNG/Halo02/params_ics.txt 1
