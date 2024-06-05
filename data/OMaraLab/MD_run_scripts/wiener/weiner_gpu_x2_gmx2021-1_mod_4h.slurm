#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=20000
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:tesla:2



## General-purpose resubmit script for GROMACS jobs on Wiener

## this runs jobs in four hour blocks with checkpinting and resubmission, 
## edit $GMXMDRUN to vary the runtime
## nsteps is set in the mdp
## Starting structure, .top, .ndx and .mdp should have the same name as the
## script, and all be in the same folder. Output will also have 
## the same name.
## Eg:  if script name is GlyT2_POPC_CHOL_r1, mdp is GlyT2_POPC_CHOL_r1.mdp


#####  SETUP ENVIRONMENT  #####

# Define error function so we can see the error code given when something
# important crashes
errexit ()
{
    errstat=$?
    if [ $errstat != 0 ]; then
        # A brief nap so slurm kills us in normal termination
        # Prefer to be killed by slurm if slurm detected some resource excess
        sleep 5
        echo "Job returned error status $errstat - stopping job sequence $SLURM_JOB_NAME at job $SLURM_JOB_ID"
        exit $errstat
    fi
}

module purge

module load gnu7/7.3.0
module load cuda/9.1.85.3
module load mvapich2/2.2
module load gromacs/2021.1


module list

export OMP_NUM_THREADS=7
export MV2_ENABLE_AFFINITY=0


GMX='mpirun -n 1 gmx_mpi'
GMXMDRUN='mpirun -n 4 gmx_mpi mdrun -maxh 3.95 -nb gpu'

#####  START MD WITH A GROMPP  #####

## GROMPP if there's no TPR file (eg, this is the first submission)
if [ ! -f ${SLURM_JOB_NAME}.tpr ]; then
    $GMX grompp -f ${SLURM_JOB_NAME}.mdp -c ${SLURM_JOB_NAME}_start.gro -o ${SLURM_JOB_NAME}.tpr -p ${SLURM_JOB_NAME}.top  -n ${SLURM_JOB_NAME}.ndx -maxwarn 1 &> ${SLURM_JOB_NAME}_grompp_${SLURM_JOB_ID}.txt
fi

#####  RUN MD FOR 3.95 HOURS  #####

$GMXMDRUN -v -deffnm ${SLURM_JOB_NAME} -cpi ${SLURM_JOB_NAME}.cpt || errexit

#####  CHECK IF JOB IS DONE; IF NOT DONE RESUBMIT THIS SCRIPT  #####

# Check the log file for the number of steps completed
steps_done=`perl -n -e'/Statistics over (\d+) steps using (\d+) frames/ && print $1' ${SLURM_JOB_NAME}.log`
# Check the mdp file for the number of steps we want
steps_wanted=`perl -n -e'/nsteps\s*=\s*(\d+)/ && print $1' ${SLURM_JOB_NAME}.mdp`
# Resubmit if we need to
if (( steps_done < steps_wanted )); then
    echo "Job ${SLURM_JOB_NAME} terminated with ${SLURM_JOB_NAME}/${SLURM_JOB_NAME} steps finished." 
    echo "Submitting next job in sequence ${SLURM_JOB_NAME}."
    sbatch ${SLURM_JOB_NAME}.sh
fi



## clean up linebreaks in output file

sed -i 's|\r|\n|g' slurm-${SLURM_JOB_ID}.out

## backup logfile for debugging

cp ${SLURM_JOB_NAME}.log lastlog.txt

echo $SLURM_JOB_NODELIST  # print node id to the end of output file for benchmarking / identifying troublesome nodes
echo "CLUSTERID=Weiner GPU" # print which cluster was used, so if a system is run on multiple systems I can separate the log files for benchmarking

#####  END  #####
