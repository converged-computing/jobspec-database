#!/bin/bash -l

#SBATCH --nodes=4
#SBATCH --partition=debug

module swap PrgEnv-cray PrgEnv-gnu
module load dask
module load jupyter/1.0.0


# get tunneling info
export XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
gateway=${EPROXY_LOGIN}
submit_host=${SLURM_SUBMIT_HOST}
port=8889

echo $node on $gateway pinned to port $port
# print tunneling instructions jupyter-log
echo -e "
To connect to the compute node ${node} on Shaheen running your jupyter notebook server,
you need to run following two commands in a terminal

1. Command to create ssh tunnel from you workstation/laptop to cdlX:
ssh -L ${port}:localhost:${port} ${user}@${submit_host}.hpc.kaust.edu.sa

2. Command to create ssh tunnel to run on cdlX:
ssh -L ${port}:${node}:${port} ${user}@${gateway}

Copy the link provided below by jupyter-server and replace the nid0XXXX with localhost before pasting it in your browser on your workstation/laptop. Do not forget to close the notebooks you open in you browser and shutdown the jupter client in your browser for gracefully exiting this job or else you will have to mannually cancel this job running your jupyter server.
"

echo "Starting dask server in background with requested resouce"
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
srun dask-mpi --no-nanny --nthreads 1  --local-directory=/project/k01/shaima0d/tickets/28242/workers${SLURM_JOBID} --scheduler-file=scheduler_${SLURM_JOBID}.json --interface=ipogif0 --scheduler-port=6192 &

# Run Jupyter
jupyter notebook --no-browser --port=${port} --ip=${node} 


