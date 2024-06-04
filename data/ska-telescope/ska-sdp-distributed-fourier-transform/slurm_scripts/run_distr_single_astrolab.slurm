#!/bin/bash
! Dask job script for AstroLab cluster
#!
#!#############################################################
#! job name
#SBATCH --job-name DFFT
#! account
#SBATCH --account astro
#! number of nodes
#SBATCH --nodes=3
#! total task numbers
#SBATCH --ntasks=3
#! memory limit
##SBATCH --mem 100M
#! time limit
#SBATCH --time=00:00:00
#! e-mail
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=$USER@cnlab.net
#! no requeue
##SBATCH --no-requeue
#! partition of nodes
#SBATCH --partition astro-cpu
#! cluster name
#SBATCH --clusters astrolab
#! max number of switches
#SBATCH --switches=1
#! hostname list of nodes
##SBATCH --nodelist astrolab-hpc-[1-13]
#! exclude node list
##SBATCH --exclude astrolab-hpc-1,astrolab-hpc-12,astrolab-hpc-13
#! std output and std error
#SBATCH --output /home/%u/work/slurm-scripts/slurm_out/slurm-std-%A.out
#SBATCH --error /home/%u/work/slurm-scripts/slurm_out/slurm-err-%A.out
#! exclusive node
#SBATCH --exclusive
#! number cpus per task
##SBATCH --cpus-per-task 1
#! allocate gpu
##SBATCH --gres=gpu
#! number of cpus per gpu
##SBATCH --cpus-per-gpu

#Define project path
project_path=${HOME}/work/ska-sdp-distributed-fourier-transform
dask_config_path=${HOME}/work/slurm-scripts/dask_config_distrim_hight_split_mini.yaml
env_name=rascil_new
slurm_worker_log_path=${HOME}/work/slurm-scripts/slurm_out/worker_log

dask_port=9900
dask_dashboardport=9901

source ${HOME}/.bashrc

# We can use the conda env infor from .bashrc, or activate env here
# env_name=rascil_new
#conda activate ${env_name}

export DASK_CONFIG=${dask_config_path}

#！setup PYTHONPATH
export PYTHONPATH=${project_path}:$PYTHONPATH

cd ${project_path}

echo ${SLURM_JOB_NODELIST}

#! run dask scheduler
localdir=/mnt/dask_tmp/$USER
srun mkdir -p ${localdir}
protool=tcp
scheduler=$(ip addr show ib0 | grep inet | grep -v inet6 | awk '{print $2}' | cut -c -12)

dask-scheduler --dashboard-address $dask_dashboardport --protocol ${protool} --interface ib0 --port $dask_port &

sleep 5

#! run dask workers
srun -o ${slurm_worker_log_path}/srun_%x_%j_worker_%n.out dask-worker --nprocs 3 --nthreads 1 --interface ib0 --memory-limit 320GiB --local-directory ${localdir} ${protool}://${scheduler}:${dask_port} &
sleep 5

echo "[main]project path : ${project_path}"
echo -e "[main]Running python: `which python`"
echo -e "[main]which main in node: `hostname`"
echo -e "[main]Running dask-scheduler: `which dask-scheduler`"
echo "[main]dask-scheduler started on ${protool}://${scheduler}:${dask_port},dashboard: http://${scheduler}:${dask_dashboardport}"
echo -e "[main]Changed directory to `pwd`.\n"

export DASK_SCHEDULER=${protool}://${scheduler}:${dask_port}

#! execute the python command here
#! This example is for demo_api
python scripts/demo_api.py --swift_config 4k[1]-n2k-512 --queue_size 300 --lru_forward 3 --lru_backward 4

