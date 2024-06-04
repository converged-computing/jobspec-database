#!/bin/sh
#PBS -l select=1:ncpus=40:ngpus=8:mpiprocs=8
#PBS -l walltime=02:00:00
#PBS -q dgx
#PBS -P 21170158
#PBS -N ML
#PBS -j oe
cd "$PBS_O_WORKDIR" || exit $?

export SINGULARITYENV_LD_LIBRARY_PATH=/home/users/astar/scei/jamesche/mylib/openmpi-1.10.7-cuda9-gnu4/lib:$SINGULARITYENV_LD_LIBRARY_PATH

export SINGULARITYENV_OMP_NUM_THREADS=4

/opt/singularity/bin/singularity exec --nv -B /home/projects/41000001/jamesche/imagenet-data:/bazel
/home/users/astar/scei/jamesche/dgx1/aitest/ub1604-python2-horovod-gpu.simg \
mpirun -np 8 \
-bind-to none -map-by slot \
-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
-mca pml ob1 -mca btl ^openib \
python
/home/users/astar/scei/jamesche/dgx1/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--model resnet50 \
--batch_size 64 \
--variable_update horovod \
--data_dir /bazel \
--data_name imagenet \
--num_batches=100