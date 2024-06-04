#!/bin/sh

### The following requests a single GPU
#PBS -l select=1:ncpus=5:ngpus=1

### The "select=1" specifies the number of nodes
### To ask for a whle node add "ncpus=40:ngpus=8"
### If you request less than 8 GPU then make the ncpus value
###   five times the ngpus value, e.g. select=1:ncpus=5:ngpus=1

### Specify amount of time required (maximum 24 hours)
#PBS -l walltime=2:00:00

### Select correct queue
#PBS -q dgx

### Specify project code
#PBS -P 99999999

### Specify name for job
#PBS -N conda-install

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### Start of commands to be run

# The NVIDIA NGC PyTorch images use Anaconda
# This examples installs OpenCV inside the Docker container using conda install

# Docker image to use for container
#   To see available images run command: nscc-docker images
#   If image is not present, email help@nscc.sg to request pulling image into repository on all DGX nodes
image="nvcr.io/nvidia/pytorch:latest"

# Change to directory where job was submitted
cd "$PBS_O_WORKDIR" || exit $?

# The "nscc-docker run $image" command runs the following Docker command: 
#    nvidia-docker run -u UID:GID --group-add PROJECT \
#      -v /home:/home -v /scratch:/scratch -v /raid:/raid \
#      --rm -i --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 $image /bin/sh
# See full list of options with "nscc-docker run -h"
# Pass the commands that you wish to run inside the container on the standard input

# When using PyTorch the --ipc=host docker option allows the container to use all the system memory as shared memory (default is 1GB)
nscc-docker run --ipc=host $image << EOF > stdout.$PBS_JOBID 2>stderr.$PBS_JOBID
hostname
nvidia-smi
pwd
# If using the <<EOF syntax then variables are expanded immediately before passing to docker.
# Therefore PBS_O_WORKDIR in the following line will be taken from the setting in the job shell
cd "$PBS_O_WORKDIR"
pwd

# protect environment values to use values from inside the container
echo PATH in job shell
echo PATH=$PATH
echo PATH inside container
echo PATH=\$PATH

# /opt/conda is read-only
# /workspace is read/write and so clone conda environment there
_conda_env=/workspace/conda-env

echo
echo clone whole base environment to \$_conda_env
conda create --clone=base -q -y -p \$_conda_env
## sometimes it is OK to use "conda create -q -y -p ... package1 package2"
## in this case that broke the dependencies for PyTorch in the base package

echo
echo activate cloned conda environment
source activate \$_conda_env

echo
echo  install additional packages
conda install -q -y opencv

echo
echo print conda information 
conda info
conda list

echo
echo Test OpenCV
python cv2-version.py
python cv2-resize.py
echo
echo  Test PyTorch
python torch-version.py
python torch-checkdevice.py

# conda environment is destroyed at end of run when container is destroyed
EOF
