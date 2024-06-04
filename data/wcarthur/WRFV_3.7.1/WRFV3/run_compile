#!/bin/bash

module purge
module load pbs
module load dot
module load intel-fc/12.1.9.293
module load intel-cc/12.1.9.293
module load openmpi/1.6.3
module load netcdf/4.3.3.1
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
export JASPERINC=/usr/include
export JASPERLIB=/usr/lib64

export WRF_EM_CORE=1

# Un-comment to compile with chemistry
#export WRF_CHEM=1

./configure

qsub -N wrf_compile << EOF_compile
#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l mem=5GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -q express
#PBS -l wd
#PBS -l software=intel-fc

module purge
module load pbs
module load dot
module load intel-fc/12.1.9.293
module load intel-cc/12.1.9.293
module load openmpi/1.6.3
module load netcdf/4.3.3.1
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
export JASPERINC=/usr/include
export JASPERLIB=/usr/lib64

# Un-comment to compile with chemistry
#export WRF_CHEM=1

export WRF_EM_CORE=1

export J="-j \$PBS_NCPUS"

((./compile em_real) 3>&2 2>&1 1>&3 | tee compile.err) 2>&1 | tee compile.log

EOF_compile

