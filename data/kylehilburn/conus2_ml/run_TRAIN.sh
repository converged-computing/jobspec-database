#!/bin/bash -l
#SBATCH --account=rda-goesstf
#SBATCH --partition=fge
#SBATCH --time=00:15:00
#SBATCH --ntasks=1

##SBATCH --nodes=1
##SBATCH --exclusive
##SBATCH --ntasks-per-node=8

##SBATCH --time=00:30:00
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=kyle.hilburn@colostate.edu

date

# use scratch for working directory so I don't exceed my home directory quota if core dumped
# make sure to stage_code.sh before running this script!
cd /scratch1/RDARCH/rda-goesstf/conus2/Code

#echo 'SLURM_NTASKS='$SLURM_NTASKS

#module load intel
#module load impi
#module use /scratch2/BMC/gsd-hpcs/bass/modulefiles
#module load openmpi
#module load cuda

#export LD_LIBRARY_PATH="/scratch1/RDARCH/rda-goesstf/cudalib/cuda-10.1-v7.6.5.32/lib64/:${LD_LIBRARY_PATH}"
#PYPATH=/scratch1/RDARCH/rda-goesstf/anaconda/bin/

MYPY=/scratch1/RDARCH/rda-goesstf/anaconda/bin/python

#srun -n $SLURM_NTASKS $MYPY MAIN_TRAIN_and_SAVE_MODEL.py configuration.txt

# requires command line argument: my_file_prefix
srun -n $SLURM_NTASKS $MYPY MAIN_TRAIN_and_SAVE_MODEL.py $1

#srun -n $SLURM_NTASKS $PYPATH/python MAIN_TRAIN_and_SAVE_MODEL.py configuration.txt
#srun -n $SLURM_NTASKS $PYPATH/python MAIN_TRAIN_and_SAVE_MODEL.py /home/Kyle.Hilburn/goesstf/conus2/configuration.txt
#mpirun -bind-to none -map-by slot -n $SLURM_NTASKS $PYPATH/python MAIN_TRAIN_and_SAVE_MODEL.py /home/Kyle.Hilburn/goesstf/conus2/configuration.txt

date
