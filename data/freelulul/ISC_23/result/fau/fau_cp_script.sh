#!/bin/bash
#SBATCH -J s_try
#SBATCH --time=00:10:00

#SBATCH -N 4

#echo ${SLURM_JOB_NODELIST}
#echo start on $(date)
export PATH=/home/hpc/b154dc/b154dc23/QE/qe-intel/qe-7.2/build-elpa/bin:$PATH

source /home/hpc/b154dc/b154dc23/spack/share/spack/setup-env.sh

spack load intel-oneapi-mkl%gcc
spack load intel-oneapi-mpi

_module_raw ()
{
    eval `/usr/bin/tclsh /apps/modules/modulecmd.tcl bash "$@"`;
    _mlstatus=$?;
    return $_mlstatus
}

module ()
{
    _module_raw "$@" 2>&1
}

mpiexec -n 256 -ppn 64 -genv I_MPI_PIN_DOMAIN=socket -genv I_MPI_PIN_ORDER=spread cp.x -pd .true. -inp cp.in
#echo end on $(date)
