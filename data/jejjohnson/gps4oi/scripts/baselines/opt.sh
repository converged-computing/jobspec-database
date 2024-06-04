#!/bin/bash

#OAR --name oi_opt
#OAR --project pr-data-ocean
#OAR -l /nodes=1,gpu=1,walltime=12:00:00
#OAR --stdout /bettik/johnsonj/logs/gps4oi_opt.log
#OAR --stderr /bettik/johnsonj/errs/gps4oi_opt.log
#OAR -e /home/johnsonj/.ssh/id_my_job_key
#OAR -p gpumodel='V100'

source /applis/environments/conda.sh
conda activate oi_torch_gpu_py39
cd /bettik/johnsonj/projects/gps4oi

# get tunneling info
XDG_RUNTIME_DIR=""
conda_env=$(which conda)
python_env=$(which python)
DIR=$(pwd)
node=$(hostname -s)
user=$(whoami)
#TODO JOB ID

cluster="bigfoot"


# print tunneling instructions jupyter-log
echo -e "
# Tunneling Info
node=${node}
user=${user}
cluster=${cluster}
port=${port}
python=${python_env}
conda=${conda_env}
dir=${DIR}
"

echo -e "
starting job!
"


python -u scripts/baselines/opt.py

echo -e "
finished job!
"
