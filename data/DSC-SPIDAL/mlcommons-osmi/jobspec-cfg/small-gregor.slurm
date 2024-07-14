#!/bin/bash

### how to run sbatch small-gregor.slurm

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-12:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu


# do in shell: time singularity pull docker://tensorflow/serving:latest-gpu
# This will produce in the dir you call it serving_latest-gpu.sif

module load anaconda
conda activate OSMI

cd /project/bii_dsc_community/tma5gv/osmi/osmi-bench/model
time python train.py $MODE

# where do you get the tensorflow-serving_latest.sif

cd /project/bii_dsc_community/tma5gv/osmi/osmi-bench/benchmark
pip install --user  -r mlcommons-osmi/../../../requirements-rivanna.txt
singularity run --nv --home `pwd` ../serving_latest-gpu.sif tensorflow_model_server --port=8500 --rest_api_port=0 --model_config_file=models.conf >& log &

sleep 12 # thats not cool
# missing is a test that shows if tensorflow model server is running ; while loop in a py prg and you leave it once ts runng
nvidia-smi

# TRAINING
MODE=small_lstm
PARAMETER=-b 32 -n 48
# MODE=medium_cnn
# MODE=large_tcnn

time python tfs_grpc_client.py -m $MODE $PARAMETER localhost:8500





