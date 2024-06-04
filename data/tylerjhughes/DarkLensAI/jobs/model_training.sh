#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=jobs/running/myjob-%A-%a.out
#SBATCH --error=jobs/running/myjob-%A-%a.err
#SBATCH --time=14:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tjhughes@swin.edu.au
#SBATCH --mail-type=ALL
#SBATCH --array=0-2

module load cuda/11.7.0
module load gcc/11.3.0
module load openmpi/4.1.4
module load python/3.10.4
module load tensorflow/2.11.0-cuda-11.7.0

# Actiate the virtual environment
PYENV_NAME=tensorflow

# Export site-packages to python path
export LD_LIBRARY_PATH=/fred/oz149/Tyler/pyenv/$PYENV_NAME/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/fred/oz149/Tyler/pyenv/$PYENV_NAME/lib/python3.7/site-packages:${PYTHONPATH}
source /fred/oz149/Tyler/pyenv/$PYENV_NAME/bin/activate
echo $PYTHONPATH

# If the job is part of an array then create a subdirectory for the array job
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p jobs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/$SLURM_ARRAY_TASK_ID
    OUT_DIR=jobs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/$SLURM_ARRAY_TASK_ID
    JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
    OUT_FILE=myjob-${SLURM_JOB_ID}.out
    ERR_FILE=myjob-${SLURM_JOB_ID}.err
fi

# If the job is not part of an array create the job directory
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    mkdir -p jobs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
    OUT_DIR=jobs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
    JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}
    OUT_FILE=myjob-${SLURM_JOB_ID}.out
    ERR_FILE=myjob-${SLURM_JOB_ID}.err
fi

# Extract the name of the submission script
SCRIPT_NAME="${SLURM_JOB_NAME}.sh"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Copy the submission script to the job directory
cp jobs/"$SCRIPT_NAME" $OUT_DIR

# Run the model_training.py script
echo "Running model_training.py"
PYTHON_SCRIPT='src/model_training.py'
DATASET_DIR="data/"
LEARNING_RATE=(0.001 0.0001 0.00001)
BATCH_SIZE=1024
LENS_MODEL="epl"

cp "$PYTHON_SCRIPT" $OUT_DIR
python -u "$PYTHON_SCRIPT" $DATASET_DIR ${LEARNING_RATE[$SLURM_ARRAY_TASK_ID]} $BATCH_SIZE $OUT_DIR $LENS_MODEL

# move the output files to the job directory
mv jobs/running/$OUT_FILE $OUT_DIR
mv jobs/running/$ERR_FILE $OUT_DIR