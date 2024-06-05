#!/bin/bash -l

#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gtx1070:1
#SBATCH --job-name=jbaik
#SBATCH --output=slurmctld.%j.out
#SBATCH --error=slurmctld.%j.err

eval "$(pyenv init -)"
export NCCL_DEBUG=INFO
export MASTER_ADDR="172.30.1.237"
export MASTER_PORT="23456"

. .slackbot

srun -o slurmd.%j.%t.out -e slurmd.%j.%t.err --export=ALL --network="MPI,DEVNAME=bond0" \
  python batch_train.py deepspeech_ctc \
    --use-cuda \
    --log-dir logs_20180906_deepspeech_ctc_h512_l4 \
    --slack \
    --visdom \
    --visdom-host 172.26.15.44 \
    --continue-from logs_20180906_deepspeech_ctc_h512_l4/deepspeech_ctc_epoch_024.pth.tar
