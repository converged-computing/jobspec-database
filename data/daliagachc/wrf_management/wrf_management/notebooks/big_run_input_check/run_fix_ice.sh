#!/bin/bash
#SBATCH -e ./error%j.txt
#SBATCH -o ./output%j.txt
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -p test
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END
#SBATCH --mail-user=diego.aliaga@helsinki.fi

# first set the environemt
export NETCDF=/appl/opt/netcdf4/gcc-7.3.0/intelmpi-18.0.2/4.6.1/
module purge
module load gcc/7.3.0  intelmpi/18.0.2 hdf5-par/1.8.20 netcdf4/4.6.1
export WRFIO_NCD_LARGE_FILE_SUPPORT=1

#exe_real=real.exe
conda activate b36
## run the executable
srun jupyter nbconvert --ExecutePreprocessor.timeout=600 --execute --to notebook --inplace check_lowinp_02.ipynb
