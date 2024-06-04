#!/bin/bash

#SBATCH --job-name=jupyterAE
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=1gb
#SBATCH -t 00-08:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=/home/lgaudet/slurm_error_logs/error_%j.err
#SBATCH --output=/home/lgaudet/slurm_output_logs/log_%j.out

#to specify specific GPU devices change gpus= to -e NVIDIA_VISIBLE_DEVICES=4,5,6

docker run -v /home/aevans:/home/aevans -v /raid/NYSM:/home/aevans/NYSM --name=jupyter_ae -w /home/aevans -u aevans --runtime=nvidia --gpus=1 -p 8886:88
88 akurbanovas/ae.ai2es:v0.1 /opt/conda/bin/jupyter lab --port=8888 --ip=0.0.0.0 --allow-root --no-browser /home/aevans

#docker run -v /raid/lgaudet:/home/lgaudet -v /raid/NYSM:/home/lgaudet/NYSM -v /rdma/flash/AI2ES/data:/home/lgaudet/model-data --name=jupyter_lg -w /home/lgaudet -u lgaudet --runtime=nvidia --gpus=1 -p 8886:88
88 akurbanovas/lg.ai2es:v0.2 python job.py
