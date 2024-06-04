#!/bin/sh

### Run in the shared test and development area
### Note that you do not have exclusive access to the GPUs in the dgx-dev queue
#PBS -q dgx-dev
#PBS -l walltime=8:00:00

### Specify project code
#PBS -P 99999999

### Specify name for job
#PBS -N jupyter

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### Start of commands to be run

# Change directory to where job was submitted
cd $PBS_O_WORKDIR || exit $?

# Docker image to use for container
#   To see available images run command: nscc-docker images
#   If image is not present, email help@nscc.sg to request pulling image into repository on all DGX nodes
image="nvcr.io/nvidia/tensorflow:latest"

# Change this number, port 8888 is default and may clash with other users
PORT=8888

# Run using host networking so that it is easier to access the port
# Modify port value if it clashes with other users
nscc-docker run --net=host $image << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID
# We can do the following if using "<<EOF" to pass variable as it is expanded by the job shell
cd $PBS_O_WORKDIR
hostname
#
# install any python packages required
export PYTHONUSERBASE=/workspace/.local
# Escaping PYTHONUSERBASE in the next line so that it is expanded inside docker and take the value set inside the container
mkdir -p \$PYTHONUSERBASE/bin
export PATH=\$PATH:\$PYTHONUSERBASE/bin
## pip install --user -q -U somepythonpackage

# Launch JupyterLab, listen on all interfaces on port $PORT
export TERM=ansi
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT \
 --NotebookApp.terminado_settings='{"shell_command": ["/bin/bash"]}' \
 --NotebookApp.allow_remote_access=True --FileContentsManager.delete_to_trash=False

EOF

### Notebook is now running on compute node
### However you cannot directly access port
### There are two methods that you can use
### 1. ssh port forwarding
### 2. reverse proxy (FRP, etc.)

### Using reverse proxies is a security risk, user is responsible for any data loss or unauthorized access.

### ssh port forwarding
### On local machine use ssh port forwarding to tunnel to node and port where job is running:
###   ssh -L$PORT:$HOST:$PORT aspire.nscc.sg     ### e.g. ssh -L8888:dgx4106:8888 aspire.nscc.sg
### On local machine go to http://localhost:$PORT and use token from found in file stderr.$PBS_JOBID
### Alternatively, pass a pre-determined token using --NotebookApp.token=... (visible to ps command on node)

