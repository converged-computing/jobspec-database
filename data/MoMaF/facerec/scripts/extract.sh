#! /bin/bash -l

#SBATCH --partition=small-g
#SBATCH --account=project_462000139
#SBATCH --job-name=face_extract
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --time=03:00:00
#SBATCH --array=0-99

## SBATCH --exclude=./exclude-list.txt

SAVE_EVERY=5
OUT_PATH=out
NO_IMAGES=""
#NO_IMAGES="--no-images"

if [[ $# == 0 ]]; then
    echo $0 : video file name argument missing
    exit 1
fi

if [[ $# > 1 ]]; then
    echo $0 : too many arguments
    exit 1
fi

echo Running in `hostname` $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $1

#. ./venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
module use /appl/local/csc/modulefiles
module load tensorflow
export PYTHONUSERBASE=/scratch/project_462000139/jorma/momaf/github/facerec/python_base

python3 -u ./facerec/extract.py \
    --n-shards $SLURM_ARRAY_TASK_COUNT \
    --shard-i $SLURM_ARRAY_TASK_ID \
    --save-every $SAVE_EVERY \
    --out-path $OUT_PATH \
    $NO_IMAGES \
    $1

if [[ $? -ne 0 ]]
then
    echo FAILED in `hostname` $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $1
    exit 1
fi

echo SUCCESS $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $1

#seff $SLURM_JOB_ID

sacct -P -n -a --format JobID,User,Group,State,Cluster,AllocCPUS,REQMEM,TotalCPU,Elapsed,MaxRSS,ExitCode,NNodes,NTasks -j $SLURM_JOB_ID

