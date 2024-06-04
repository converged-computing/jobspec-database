#!/bin/bash
#SBATCH -e ./error%j.txt
#SBATCH -o ./output%j.txt
#SBATCH -J flex
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -p serial
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=END
#SBATCH --mail-user=diego.aliaga@helsinki.fi

# first set the environemt
export NETCDF=/appl/opt/netcdf4/gcc-7.3.0/intelmpi-18.0.2/4.6.1/
module purge
module load gcc/7.3.0  intelmpi/18.0.2 hdf5-par/1.8.20 netcdf4/4.6.1
export WRFIO_NCD_LARGE_FILE_SUPPORT=1

#flex_dir='/homeappl/home/aliagadi/appl_taito/flexpart/Src_flexwrf_v3.3.2-omp/examples'
flex_dir='/homeappl/home/aliagadi/appl_taito/FLEXPART-WRF_v3.3.2'
input_flex=/homeappl/home/aliagadi/wrk/DONOTREMOVE/flexpart_management_data/runs/run_2020-01-03_19-44-36_/2018-02-19/flx_input
cd $flex_dir
#exe=flexwrf33_gnu_mpi
#exe=flexwrf33_gnu_omp
#exe=flexwrf33_gnu_serial
exe=flexwrf33_gnu_serial
## run my MPI executable
srun $exe $input_flex
