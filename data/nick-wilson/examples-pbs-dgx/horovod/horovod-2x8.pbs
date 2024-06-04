#!/bin/bash
### Run across 2 nodes
#PBS -l select=2:ncpus=40:ngpus=8:mpiprocs=8
#PBS -l walltime=1:00:00
#PBS -q dgx

### Specify project code
#PBS -P 99999999

#PBS -N horovod
#PBS -j oe

# Tensorflow Resnet Imagenet

# Number of layers in CNN
layers=152
# batch size
batch=64
# Filesystem holding dataset: Lustre, NFS or SSD
fs="lustre"

# Singularity image to use
image="$SINGULARITY_IMAGES/nvcr.io/nvidia/tensorflow:19.07-py3.simg"

#dataset
dataset=resnet

# Change to directory where job was submitted
if [ x"$PBS_O_WORKDIR" != x ] ; then
 cd "$PBS_O_WORKDIR" || exit $?
fi

if [ x"$fs" == xlustre ] ; then
 # dataset in Lustre
 data="/scratch/projects/ai/datasets/imagenet/$dataset"
elif [ x"$fs" == xnfs ] ; then
 # dataset in NFS
 data="/home/projects/ai/datasets/imagenet/$dataset"
else
 # dataset in local SSD
 data="/raid/datasets/$dataset"
fi

### If your MPI is not tighly integrated with PBS
### then uncomment the next line to trigger cgroups hook
#pbsdsh hostname

# Location of MPI built using "./configure --with-tm=/opt/pbs"
PATH="/home/app/dgx/openmpi-3.1.3-gnu/bin:$PATH" ; export PATH

# echo quick check of MPI
echo =====
mpicc -o mpi-hello ../mpi/mpi-hello.c
mpirun ./mpi-hello
mpicc -o mpi_nx2latbw ../mpi/mpi_nx2latbw.c
mpirun ./mpi_nx2latbw
echo =====

# Main run
SINGULARITYENV_NCCL_DEBUG=INFO ; export SINGULARITYENV_NCCL_DEBUG
mpirun \
 -x SINGULARITYENV_NCCL_DEBUG \
 --mca btl_openib_warn_default_gid_prefix 0 \
 /opt/singularity/bin/singularity exec --nv $image \
  python /workspace/nvidia-examples/cnn/resnet.py \
   --layers $layers \
   -b $batch \
   -i 200 \
   -u batch \
   --data_dir $data
