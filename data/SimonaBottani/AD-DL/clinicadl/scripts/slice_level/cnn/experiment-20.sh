#!/bin/bash
#SBATCH --partition=gpu_p1
#SBATCH --time=90:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --mem=60G
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1        # on réserve des coeurs physiques et non logiques
#SBATCH --ntasks=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs/AD-DL/train/slice_level
#SBATCH --output=./exp20/pytorch_job_%j.out
#SBATCH --error=./exp20/pytorch_job_%j.err
#SBATCH --job-name=exp20_cnn
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --mail-type=END
#SBATCH --mail-user=mauricio.diaz@inria.fr

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment training autoencoder
eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

SCRIPT="train_CNN_bad_data_split.py"

# Network structure
NETWORK="resnet18"
COHORT="ADNI"
DATE="reproducibility_results_2"

# Input arguments to clinicadl
CAPS_DIR="$SCRATCH/../commun/datasets/${COHORT}_rerun"
TSV_PATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="$SCRATCH/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=32
GPU=1

# Dataset Management
PREPROCESSING='linear'
DIAGNOSES="AD CN"
MRI_PLANE=0
SPLITS=5
SPLIT=$SLURM_ARRAY_TASK_ID

# Training arguments
EPOCHS=50
BATCH=32
BASELINE=1
LR=1e-6
WEIGHT_DECAY=1e-4
PATIENCE=15
TOLERANCE=0

# Other options
OPTIONS=""

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi


if [ $BASELINE = 1 ]; then
echo "using only baseline data"
OPTIONS="$OPTIONS --baseline"
fi

NAME="slice2D_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-AD-CN_baseline-${BASELINE}_preparedl-1_bad_split"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
NAME="${NAME}_splits-${SPLITS}"
fi

echo $NAME

# Run clinicadl
python $HOME/code/AD-DL/clinicadl/clinicadl/slice_level/$SCRIPT \
  $CAPS_DIR \
  $TSV_PATH \
  $OUTPUT_DIR$NAME \
  --network $NETWORK \
  --batch_size $BATCH \
  --diagnoses $DIAGNOSES \
  --mri_plane $MRI_PLANE \
  --n_splits $SPLITS \
  --split $SPLIT \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --patience $PATIENCE \
  --tolerance $TOLERANCE \
  --prepare_dl \
  $OPTIONS
