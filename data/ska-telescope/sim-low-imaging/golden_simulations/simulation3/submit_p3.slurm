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
#SBATCH --mem 50000
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

scheduler=10.60.253.22
echo "run dask-scheduler"
ssh $host dask-scheduler --port=8786 --local-directory /mnt/storage-ssd/tim/dask-workspace &
sleep 5

hostIndex=0
for host in `cat hostfile.$JOBID`; do
    echo "Working on $host ...."
    echo "run dask-worker"
    ssh $host dask-worker --nprocs 1 --nthreads 1 --memory-limit 100GB \
        --interface ib0 --local-directory /mnt/storage-ssd/tim/dask-workspace $scheduler:8786  &
        sleep 1
    hostIndex="1"
done
echo "Scheduler and workers now running"

#! We need to tell dask Client (inside python) where the scheduler is running
export ARL_DASK_SCHEDULER=${scheduler}:8786
echo "Scheduler is running at ${scheduler}"

cp ../../clean_ms.py .

CMD="python ../../clean_ms.py --ngroup 2 --nworkers 0 --weighting uniform --context wprojection \
--mode invert --niter 1000 --nmajor 3 --fractional_threshold 0.2 --threshold 0.01 --nmoment 1 \
--amplitude_loss 0.25 --deconvolve_facets 8 --deconvolve_overlap 32 \
--msname /mnt/storage-ssd/tim/Code/sim-low-imaging/data/EoR0_20deg_24.MS \
--time_coal 0.0 --frequency_coal 0.0 --channels 131 146 --window_shape no_edge --window_edge 16 \
--use_serial_invert True --use_serial_predict True --plot False --fov 2.0 --single False | tee clean_ms.log"

echo "About to execute $CMD"

eval $CMD

