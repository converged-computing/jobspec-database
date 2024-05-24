#!/bin/bash -l
#SBATCH --job-name=GE-fortranMPI_gnu
#SBATCH --partition=test
#SBATCH --nodes=3
#SBATCH --ntasks=30
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --export=NONE

#SBATCH --account=ea007
# to not share nodes with other running jobs use #SBATCH -exclusive


# To swap from Cray Environment to GNU
module load gcc/8.3.1 openmpi/4.0.5

# leave in, it lists the environment loaded by the modules
module list

#  Note: SLURM_JOBID is a unique number for every job.
#  These are generic variables
EXECUTABLE=hello_mpi_gnu
SCRATCH=$MYSCRATCH/run_fortranMPI_gnu/$SLURM_JOBID
RESULTS=$MYGROUP/mpifortran_gnu_results/$SLURM_JOBID
#  for openMP codes
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then 
    mkdir -p $SCRATCH 
fi 
echo SCRATCH is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then 
    mkdir -p $RESULTS 
fi
echo the results directory is $RESULTS

################################################
# declare the name of the output file or log file
OUTPUT=fortranMPI_gnu.log

#############################################
#   Copy input files to $SCRATCH
#   then change directory to $SCRATCH
cd ${SLURM_SUBMIT_DIR}

cp $EXECUTABLE $SCRATCH

cd $SCRATCH
echo " "
echo " first with mpirun "
echo " "
#
#mpirun -np 10 ./$EXECUTABLE >> ${OUTPUT}
#mpirun -np ${SLURM_NTASKS} ./$EXECUTABLE >> ${OUTPUT}
#srun  -N ${SLURM_JOB_NUM_NODES}
#srun -n ${SLURM_NTASKS} ./$EXECUTABLE >> ${OUTPUT}
echo " " 
echo " okay now with srun "
echo "  "
#srun -N 1 -n 10 ./$EXECUTABLE >> ${OUTPUT}
srun --mpi=pmix_v3 ./${EXECUTABLE} >> ${OUTPUT}
#############################################
#    $OUTPUT file to the unique results dir
# note this can be a copy or move  
mv  $OUTPUT ${RESULTS}

cd $HOME

###########################
# Clean up $SCRATCH 

rm -r $SCRATCH

echo fortranMPI_gnu job finished at  `date`


