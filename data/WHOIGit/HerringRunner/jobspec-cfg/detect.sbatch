#!/usr/bin/env bash

#SBATCH --job-name=detectyolo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%j.%x.log


echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
echo "Command: $0 $@"
cd /vortexfs1/scratch/sbatchelder/herringrunner
hostname; pwd; date; echo "CUDA=$CUDA_VISIBLE_DEVICES"

module load cuda10.1/{toolkit,blas,fft,cudnn}
source herring_yolo_env/bin/activate
echo "Environment... Loaded"

MODEL=$1
DATA=$2
NAME=$3

cd yolov5_ultralytics 

time python detect.py --weights "../$MODEL" --source "../$DATA" --project "../detect-output" --name "$NAME" --save-txt --nosave --hide-labels --device $CUDA_VISIBLE_DEVICES


