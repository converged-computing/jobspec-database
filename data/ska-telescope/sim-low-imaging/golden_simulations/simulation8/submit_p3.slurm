#!/bin/bash
#!
#! Dask job script for P3
#! Tim Cornwell
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J IMAGING
#! Which project should be charged:
#SBATCH -A SKA-SDP
#! How many whole nodes should be allocated?
#SBATCH --nodes=16
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=32
#! Memory limit: P3 has roughly 107GB per node
#SBATCH --mem 100000
#! How much wallclock time will be required?
#SBATCH --time=23:59:59
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL,END
#! Where to send email messages
#SBATCH --mail-user=realtimcornwell@gmail.com
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p compute
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
module purge                               # Removes all modules still loaded

#! Set up python
# . $HOME/alaska-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ARL
echo "PYTHONPATH is ${PYTHONPATH}"

echo -e "Running python: `which python`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

cd $SLURM_SUBMIT_DIR
echo -e "Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Create a hostfile:
scontrol show hostnames $SLURM_JOB_NODELIST | uniq > hostfile.$JOBID

scheduler=$(head -1 hostfile.$JOBID)

dask-ssh --nthreads 4 --memory-limit 100GB --nprocs 2 --hostfile hostfile.$JOBID --log-directory $SLURM_SUBMIT_DIR &
sleep 5

echo "Scheduler and workers now running"

#! We need to tell dask Client (inside python) where the scheduler is running
export ARL_DASK_SCHEDULER=${scheduler}:8786
echo "Scheduler is running at ${scheduler}"

cp ../../clean_ms.py .

CMD="python ./clean_ms.py --ngroup 1 --nworkers 0 --weighting natural --context wprojection \
--mode invert --amplitude_loss 0.25 --channels 131 147 \
--msname /mnt/storage-ssd/tim/Code/sim-low-imaging/data/EoR0_20deg_192.MS \
--use_serial_invert True --use_serial_predict True --plot False --fov 2.0 --single False | tee clean_ms.log"

echo "About to execute $CMD"

eval $CMD

