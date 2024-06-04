#!/bin/bash
#PBS -P raw_audio_transformer
#PBS -j oe
#PBS -N audiotx_samplelen128_emb256_depth4
#PBS -q volta_gpu
#PBS -l select=1:ncpus=10:mem=80gb:ngpus=1
#PBS -l walltime=72:00:00

# Change to directory where job was submitted
if [ x"$PBS_O_WORKDIR" != x ] ; then
 cd "$PBS_O_WORKDIR" || exit $?
fi

D=$(date +%Y.%m.%d_%H.%M.%S) #'%(%Y-%m-%d)T'
logdir=logs/logs_$D
mkdir -p $logdir

image=/app1/common/singularity-img/3.0.0/user_img/tifresi_freesound_gpst-pytorch_1.7_ngc-20.12-py3.simg
singularity exec $image bash <<EOF > $logdir/stdout.$PBS_JOBID 2> $logdir/stderr.$PBS_JOBID

python main.py > $logdir/stdout.log

EOF