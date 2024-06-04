#!/bin/sh

### Requesting 4 GPUs on 1 DGX-1 node (ncpus=5*ngpus)
#PBS -l select=1:ncpus=20:ngpus=4

### Specify amount of time required
#PBS -l walltime=1:00:00

### Specify DGX queue
#PBS -q dgx

### Specify project code
#PBS -P 99999999

### Specify name for job
#PBS -N bmt_pytorch

### Merge standard output and error
#PBS -j oe

# Docker image to use for container
image="nscc/local/pytorch:latest"
## Number of GPUs (one MPI process per GPU)
ngpu=4
## Python script to run
script="pytorch_synthetic_benchmark.py"
## Add --ipc=host to docker command to use shared memory on host
dockerargv="--ipc=host"

# Change to directory where job was submitted
cd "$PBS_O_WORKDIR" || exit $?

#### The input file depends on the version of Horovod
## $ pip freeze | grep horovod
## horovod==0.18.2
## $ wget https://raw.githubusercontent.com/horovod/horovod/v0.18.2/examples/pytorch_synthetic_benchmark.py

# Pass commands to standard input of Docker container
nscc-docker run $dockerargv $image << EOF
cd $PBS_O_WORKDIR
mpirun --mca btl_base_warn_component_unused 0 -np $ngpu python pytorch_synthetic_benchmark.py
EOF
