#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --export=NONE


module load python/3.6.3
module load jupyter/1.0.0
module load numpy pandas h5py matplotlib scikit-image scikit-learn dask graphviz


# get tunneling info
export XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="zeus"
port=8888
bokeh_port=8889
bokeh_host=$(srun -n 1 /bin/hostname)

# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.pawsey.org.au

To visualize the Dashboard created by Dask Distributed:
ssh -N -f -L ${bokeh_port}:${bokeh_host}:${bokeh_port} ${user}@${cluster}.pawsey.org.au

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# Run Jupyter
jupyter-notebook --no-browser --port=${port} --ip=${node} &

# In the interest of keeping the output of jupyter server seperate, sleep for a little while.
sleep 20

# Run dask distributed clinets 
srun --export=all -n ${SLURM_NTASKS} -N ${SLURM_JOB_NUM_NODES} -c ${SLURM_CPUS_PER_TASK} --export=all dask-mpi --no-nanny --nthreads ${SLURM_CPUS_PER_TASK} --bokeh-port=${bokeh_port} --local-directory ${PWD}/distributed_workspace --scheduler-file ${PWD}/distributed_workspace/scheduler.json 

