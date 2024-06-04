#!/bin/bash
#OAR -n Full convolutions
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=01:00:00
#OAR --stdout full_convolutions_%jobid%.out
#OAR --stderr full_convolutions_%jobid%.err
#OAR --project fidle

#---- Note for cpu, set :
# OAR -l /nodes=1/core=32,walltime=02:00:00
# and add a 2>/dev/null to ipython xxx

# -----------------------------------------------
#         _           _       _
#        | |__   __ _| |_ ___| |__
#        | '_ \ / _` | __/ __| '_ \
#        | |_) | (_| | || (__| | | |
#        |_.__/ \__,_|\__\___|_| |_|
#                             Fidle at GRICAD
# -----------------------------------------------
#
# <!-- TITLE --> [K3GTSRB10] - OAR batch script submission
# <!-- DESC -->  Bash script for an OAR batch submission of an ipython code
# <!-- AUTHOR : Jean-Luc Parouty (CNRS/SIMaP) -->

# ==== Notebook parameters =========================================

CONDA_ENV='fidle'
NOTEBOOK_DIR="~/fidle/GTSRB"

SCRIPT_IPY="03-Better-convolutions.py"

# ---- Environment vars used to override notebook/script parameters

# 'enhanced_dir', 'model_name', 'dataset_name', 'batch_size', 'epochs', 'scale', 'fit_verbosity'

export FIDLE_OVERRIDE_GTSRB3_run_dir="./data"
export FIDLE_OVERRIDE_GTSRB3_enhanced_dir="./run/GTSRB3"
export FIDLE_OVERRIDE_GTSRB3_model_name="model_01"
export FIDLE_OVERRIDE_GTSRB3_dataset_name="set-24x24-L"
export FIDLE_OVERRIDE_GTSRB3_batch_size=64
export FIDLE_OVERRIDE_GTSRB3_epochs=5
export FIDLE_OVERRIDE_GTSRB3_scale=1
export FIDLE_OVERRIDE_GTSRB3_fit_verbosity=0

# ==================================================================

echo '------------------------------------------------------------'
echo "Start : $0"
echo '------------------------------------------------------------'
echo "Notebook dir  : $NOTEBOOK_DIR"
echo "Script        : $SCRIPT_IPY"
echo "Environment   : $CONDA_ENV"
echo '------------------------------------------------------------'
env | grep FIDLE_OVERRIDE | awk 'BEGIN { FS = "=" } ; { printf("%-35s : %s\n",$1,$2) }'
echo '------------------------------------------------------------'

source /applis/environments/cuda_env.sh dahu 10.0
source /applis/environments/conda.sh
#
conda activate "$CONDA_ENV"

# ---- Run it...
#
cd $NOTEBOOK_DIR

ipython "$SCRIPT_IPY"

echo 'Done.'
