#!/bin/bash
#PBS -P bd36
#PBS -q gpuvolta
#PBS -l walltime=04:00:00
#PBS -l mem=32GB
#PBS -l jobfs=16000MB
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l other=mpi:hyperthread
#PBS -l wd
#PBS -r y
#PBS -l storage=scratch/bd36+gdata/q95

# THIS WAS ADA TEST 01       
# GOAL:  SEE IF IT WORKS

## General-purpose resubmit script for GROMACS jobs on Bunya

## this runs jobs in four hour blocks with checkpinting and resubmission, 
## edit $GMXMDRUN to vary the runtime
## nsteps is set in the mdp
## Starting structure, .top, .ndx and .mdp should have the same name as the
## script, and all be in the same folder. Output will also have 
## the same name.
## Eg:  if script name is GlyT2_POPC_CHOL_r1, mdp is GlyT2_POPC_CHOL_r1.mdp
##
## IMPORTANT: SCRIPT NAME SHOULD NOT END WITH .sh
##
## IMPORTANT: If SCRIPTNAME.mdp does not exist, automatic resubmission will fail
##
## IMPORTANT:  YOU NEED TO HAVE ACCESS TO A GROMACS SINGULARITY CONTAINER.  

 

#####  SETUP ENVIRONMENT  #####

# Load module, always specify version number.
module load singularity
 
# Must include `#PBS -l storage=scratch/ab12+gdata/yz98` if the job
# needs access to `/scratch/ab12/` and `/g/data/yz98/`. Details on:
# https://opus.nci.org.au/display/Help/PBS+Directives+Explained
 

# Define error function so we can see the error code given when something
# important crashes
errexit ()

{
    errstat=$?
    if [ $errstat != 0 ]; then
        # A brief nap so slurm kills us in normal termination
        # Prefer to be killed by slurm if slurm detected some resource excess
        sleep 5
        echo "Job returned error status $errstat - stopping job sequence $PBS_JOBNAME at job $PBS_JOB_ID"
        exit $errstat
    fi
}


export SINGULARITY_TMPDIR=/scratch/bd23/aq8103/tmp/ 



GMX='singularity run --nv /g/data/q95/SHARED/gromacs_2022.3.sif gmx'
GMXMDRUN='singularity run --nv /g/data/q95/SHARED/gromacs_2022.3.sif gmx mdrun -ntmpi 1 -ntomp 12 -pin on -dlb yes -maxh 3.95'

#####  START MD WITH A GROMPP  #####

## GROMPP if there's no TPR file (eg, this is the first submission)
if [ ! -f ${PBS_JOBNAME}.tpr ]; then
    $GMX grompp -f ${PBS_JOBNAME}.mdp -c ${PBS_JOBNAME}_start.gro -o ${PBS_JOBNAME}.tpr -p ${PBS_JOBNAME}.top  -n ${PBS_JOBNAME}.ndx -maxwarn 2 &> ${PBS_JOBNAME}_grompp_${PBS_JOBID}.txt
fi

#####  RUN MD FOR 9.95 HOURS  #####

$GMXMDRUN -v -deffnm ${PBS_JOBNAME} -cpi ${PBS_JOBNAME}.cpt || errexit

#####  CHECK IF JOB IS DONE; IF NOT DONE RESUBMIT THIS SCRIPT  #####


# Check the log file for the number of steps completed
steps_done=`perl -n -e'/Statistics over (\d+) steps using (\d+) frames/ && print $1' ${PBS_JOBNAME}.log`
# Check the mdp file for the number of steps we want
steps_wanted=`perl -n -e'/nsteps\s*=\s*(\d+)/ && print $1' ${PBS_JOBNAME}.mdp`
# Resubmit if we need to
if (( steps_done < steps_wanted )); then
    echo "Job ${PBS_JOBNAME} terminated with ${PBS_JOBNAME}/${PBS_JOBNAME} steps finished." 
    echo "Submitting next job in sequence ${PBS_JOBNAME}."
    qsub ${PBS_JOBNAME}
fi

#####  END  #####
