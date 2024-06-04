#!/bin/bash
#SBATCH --job-name=mborrus_MiMA
#SBATCH --ntasks=32
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --constraint=[CLASS:SH3_CBASE|CLASS:SH3_CPERF]
#SBATCH --partition=serc
#SBATCH -o ./jobfiles/mima_test%j.out
#SBATCH -e ./jobfiles/mima_test%j.err

# Load modules
module purge
. /home/groups/s-ees/share/cees/spack_cees/scripts/cees_sw_setup-beta.sh

CEES_MODULE_SUFFIX="cees-beta"
COMP="intel"
MPI="mpich"

# Load intel 
module load devel gcc/10.
module load intel-${CEES_MODULE_SUFFIX}
module load mpich-${CEES_MODULE_SUFFIX}/
module load netcdf-c-${CEES_MODULE_SUFFIX}/
module load netcdf-fortran-${CEES_MODULE_SUFFIX}/

module list

conda activate wavenet_env

#overkill to make sure everything is seen...
export "PYTHONPATH=$PYTHONPATH:/home/mborrus/"
export "PYTHONPATH=$PYTHONPATH:/scratch/users/mborrus/MiMA/code/MiMAv0.1_mborrus/src/atmos_param/dd_drag/"
export "PYTHONPATH=$PYTHONPATH:/scratch/users/mborrus/MiMA/code/MiMAv0.1_mborrus/src/atmos_param/"
export "PYTHONPATH=$PYTHONPATH:/scratch/users/mborrus/MiMA/code/MiMAv0.1_mborrus/wavenet/"
export "PYTHONPATH=$PYTHONPATH:/scratch/users/mborrus/MiMA/code/MiMAv0.1_mborrus/wavenet/models/"
export "HDF5_DISABLE_VERSION_CHECK=1"

# setup run directory
run=namelist
N_PROCS=32

base=/scratch/users/mborrus/MiMA
user=mborrus
executable=${base}/code/MiMAv0.1_mborrus/exp/exec.Sherlock/mima.x
input=${base}/code/MiMAv0.1_mborrus/input
rundir=${base}/runs/$run

# Make run dir
[ ! -d $rundir ] && mkdir $rundir
# Copy executable to rundir
cp $executable $rundir/
# Copy input to rundir
cp -r $input/* $rundir/
# Run the model
cd $rundir

ulimit -s unlimited

[ ! -d RESTART ] && mkdir RESTART
srun --ntasks 32 --mem-per-cpu 8G mima.x


CCOMB=${base}/code/MiMAv0.1_mborrus/bin/mppnccombine.Sherlock
$CCOMB -r atmos_daily.nc atmos_daily.nc.*
$CCOMB -r atmos_avg.nc atmos_avg.nc.*
