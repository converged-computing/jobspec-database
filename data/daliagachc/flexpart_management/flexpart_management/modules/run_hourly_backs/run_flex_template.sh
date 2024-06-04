#!/bin/bash
#SBATCH -e ./error%j.txt
#SBATCH -o ./output%j.txt
#SBATCH -J {run_name}
#SBATCH -n {cpu_num}
#SBATCH -t {run_time}
#SBATCH -p {run_type}
#SBATCH --mem-per-cpu={run_mem}
#SBATCH --mail-type=END
#SBATCH --mail-user=diego.aliaga@helsinki.fi

# first set the environemt
export NETCDF=/appl/opt/netcdf4/gcc-7.3.0/intelmpi-18.0.2/4.6.1/
module purge
module load gcc/7.3.0  intelmpi/18.0.2 hdf5-par/1.8.20 netcdf4/4.6.1
export WRFIO_NCD_LARGE_FILE_SUPPORT=1

#. ./run_name.sh
#flex_dir='/homeappl/home/aliagadi/appl_taito/flexpart/Src_flexwrf_v3.3.2-omp/examples'
#flex_dir={flex_dir}
#input_flex="/homeappl/home/aliagadi/wrk/DONOTREMOVE/flexpart_management_data/runs/${run_name}/flex_input"
cd {flex_dir}
#exe=flexwrf33_gnu_mpi
#exe=flexwrf33_gnu_omp
#exe=flexwrf33_gnu_serial
## run my MPI executable
srun {flex_exe} {input_flex}
