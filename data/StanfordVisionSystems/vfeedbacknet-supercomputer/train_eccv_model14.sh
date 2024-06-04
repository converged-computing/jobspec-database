#!/bin/bash
#
#SBATCH --job-name=emodel14_jester
#SBATCH --output=eccv_model14_%j.log
#SBATCH --error=eccv_model14_%j.log
#
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64000M
#SBATCH --tmp=64000M
#SBATCH --gres gpu:4

date
hostname

echo -n 'loading modules ... '
module load GCC/4.9.2-binutils-2.25 
module load OpenMPI/1.8.5
module load Python/3.6.0
module load tensorflow/1.5.0-cp36
echo 'done!'

echo -n 'activating python virtualenv ... '
source $HOME/.local/venv/bin/activate
echo 'done!'

echo '---log system information---'
echo 'num_cpus' $(nproc)
nvidia-smi
df -Th
free -h
echo 'done!'

TMPFS=$TMPDIR
echo -n 'creating data dir' $TMPFS '... '
mkdir -p $TMPFS
echo 'done!'

date
echo -n 'extracting data into dir:' $TMPFS '... '
pv $WORK/datasets/20bn-jester.xs.tar | tar --skip-old-files -xf - -C $TMPFS
echo 'done!'
date

export DATA_ROOT=$TMPFS/20bn-jester

echo 'running training script'
$HOME/projects/vfeedbacknet/scripts/jemmons_train_20bn-jester.xs.sh 0,1,2,3 vfeedbacknet_eccv_model14 $WORK/vfeedbacknet-results/20bn/vfeedbacknet_eccv_model14.loss_uniform.xs --video_length=20 --video_height=112 --video_width=112 --video_downsample_ratio=2 --learning_rate_init=0.1 --learning_rate_decay=0.998 --learning_rate_min=0.001 --global_step_init 0 --train_batch_size=128 --prefetch_batch_size=1024 --validation_interval=16 --last_loss_multipler=1 --num_gpus=4 --num_cpus=5 --pretrain_root_prefix=$WORK/pretrained-models

date
echo 'finshed.'