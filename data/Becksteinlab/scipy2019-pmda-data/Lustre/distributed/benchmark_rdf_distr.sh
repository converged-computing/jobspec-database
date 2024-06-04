#!/bin/bash
#SBATCH -J PMDA_BM                  # name
#SBATCH --partition=compute
#SBATCH --nodes=3                        # Total number of nodes requested (16 cores/node). You may delete this line if wanted
#SBATCH --ntasks-per-node=24            # Total number of mpi tasks requested
#SBATCH --export=ALL
#SBATCH -t 08:00:00                      # wall time (D-HH:MM)
#SBATCH --mail-type=ALL                # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=sfan19@asu.edu  # send-to address

bash /home/sfan19/.bashrc

echo $SLURM_JOB_ID
echo $USER

SCHEDULER=`hostname`
echo SCHEDULER: $SCHEDULER
dask-scheduler --port=8786 &
sleep 5

hostnodes=`scontrol show hostnames $SLURM_NODELIST`
echo $hostnodes

for host in $hostnodes; do
    echo "Working on $host ...."
    ssh $host dask-worker --nprocs 24 --nthreads 1 $SCHEDULER:8786 &
    sleep 1
done


python benchmark_rdf_distr.py $SCHEDULER:8786
