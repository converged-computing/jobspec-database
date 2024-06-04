#!/usr/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:40:00
#PBS -N sequencing
#PBS -A cnm72801
#PBS -o job.out
#PBS -e job.err

cd $PBS_O_WORKDIR

#module purge 
#export PATH="/home/share/cnm50256/bin/miniconda2/bin:$PATH"
#export PATH="/home/share/cnm50256/bin/lammps-31Mar17-mod/bin:$PATH"
#export PATH="/home/tpatra/.local/lib/python3.6/site-packages/geneticalgorithm:$PATH"
module load intel
module load openmpi/1.10/intel-17
module load fftw3/3.3/openmpi-1.10/intel-16
module load lammps/2015/openmpi-1.10/intel-16
#module PATH="/opt/apps/python-intel/3.6.3-2018.3.039-1-2018-el6/intelpython3/tornado:$PATH"

function echoMe {
        mpirun lmp_mpi < in_b_$1.lammps > dump_$1.log
        exit 0
}
