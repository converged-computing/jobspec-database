#!/bin/bash
#PBS -l select=1:ncpus=20:ngpus=4:mpiprocs=4
#PBS -l walltime=4:00:00
#PBS -q dgx
#PBS -N horovod
#PBS -j oe
### Specify project code
#PBS -P 99999999

# Tensorflow Resnet Imagenet

# Number of layers in CNN
layers=152
# batch size
batch=64
# NFS(1) or SSD(0)
nfs=1

# Singularity image to use
image="/home/projects/ai/singularity/nvcr.io/nvidia/tensorflow:18.08-py3.simg"

#dataset
dataset=resnet

# Change to directory where job was submitted
if [ x"$PBS_O_WORKDIR" != x ] ; then
 cd "$PBS_O_WORKDIR" || exit $?
fi

if [ $nfs -eq 1 ] ; then
# dataset in NFS
data="/home/projects/41000001/datasets/imagenet/$dataset"
else
# dataset in local SSD
data="/raid/datasets/$dataset"
fi

# Number of MPI processes to launch
# Equal to number of GPUs
# set by the mpiprocs paramater in "#PBS select -l"
np=`cat $PBS_NODEFILE | wc -l`

# On a single node you can start the MPI inside the container
/opt/singularity/bin/singularity exec --nv $image /bin/bash << EOF
NCCL_DEBUG=INFO ; export NCCL_DEBUG
time mpirun -np $np \
 -x NCCL_DEBUG \
 python /workspace/nvidia-examples/cnn/resnet.py \
   --layers $layers \
   -b $batch \
   -i 2000 \
   -u batch \
   --data_dir $data
EOF
