#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=10000
#SBATCH --job-name=tb
#SBATCH --output=out/tb.out
#SBATCH --mail-user=larend@mit.edu
#SBATCH --mail-type=FAIL

if [ $HOSTNAME = dgx1 ]; then
    singularity exec --nv -B /raid:/raid /raid/poggio/home/larend/localtensorflow.img \
tensorboard --logdir=\
00000:/raid/poggio/home/larend/models/robust/cifar10/00000,\
00001:/raid/poggio/home/larend/models/robust/cifar10/00001,\
00002:/raid/poggio/home/larend/models/robust/cifar10/00002,\
00003:/raid/poggio/home/larend/models/robust/cifar10/00003,\
00004:/raid/poggio/home/larend/models/robust/cifar10/00004,\
00005:/raid/poggio/home/larend/models/robust/cifar10/00005,\
00006:/raid/poggio/home/larend/models/robust/cifar10/00006,\
00007:/raid/poggio/home/larend/models/robust/cifar10/00007,\
00008:/raid/poggio/home/larend/models/robust/cifar10/00008,\
00009:/raid/poggio/home/larend/models/robust/cifar10/00009 \
--port=${1:-6050}
elif [ $HOSTNAME = gpu-16 ] || [ $HOSTNAME = gpu-17 ] ; then
    source /cbcl/cbcl01/larend/envs/robust/bin/activate
    module add cuda/8.0
    module add cudnn/8.0-v5.1
    tensorboard --logdir=\
00000:/cbcl/cbcl01/larend/models/robust/cifar10/00000,\
00001:/cbcl/cbcl01/larend/models/robust/cifar10/00001,\
00002:/cbcl/cbcl01/larend/models/robust/cifar10/00002,\
00003:/cbcl/cbcl01/larend/models/robust/cifar10/00003,\
00004:/cbcl/cbcl01/larend/models/robust/cifar10/00004,\
00005:/cbcl/cbcl01/larend/models/robust/cifar10/00005,\
00006:/cbcl/cbcl01/larend/models/robust/cifar10/00006,\
00007:/cbcl/cbcl01/larend/models/robust/cifar10/00007,\
00008:/cbcl/cbcl01/larend/models/robust/cifar10/00008,\
00009:/cbcl/cbcl01/larend/models/robust/cifar10/00009 \
--port=${1:-6050}
else
    module load openmind/singularity/older_versions/2.4
    singularity exec --nv -B /om:/om /om/user/larend/localtensorflow.img \
tensorboard --logdir=\
00000:/om/user/larend/models/robust/cifar10/00000,\
00001:/om/user/larend/models/robust/cifar10/00001,\
00002:/om/user/larend/models/robust/cifar10/00002,\
00003:/om/user/larend/models/robust/cifar10/00003,\
00004:/om/user/larend/models/robust/cifar10/00004,\
00005:/om/user/larend/models/robust/cifar10/00005,\
00006:/om/user/larend/models/robust/cifar10/00006,\
00007:/om/user/larend/models/robust/cifar10/00007,\
00008:/om/user/larend/models/robust/cifar10/00008,\
00009:/om/user/larend/models/robust/cifar10/00009 \
--port=${1:-6050}
fi
