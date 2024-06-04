#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=32  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=160G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham. 120
#SBATCH --time=3-10:00     # DD-HH:MM:SS
#SBATCH --mail-user=sebastien.henwood@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/%x-%A-%a.out
#SBATCH --account=rrg-franlp

module load StdEnv/2020 python/3.10 cuda cudnn gcc/9.3.0 arrow

echo "User $USER on shell $0"
SOURCEDIR=~/projects/def-franlp/$USER/Samson
cd $SOURCEDIR
nvidia-smi

###
# ENV PREPARATION
# Push a cached venv
###
cp ~/projects/def-franlp/$USER/venv.tar.gz $SLURM_TMPDIR
tar xzf $SLURM_TMPDIR/venv.tar.gz -C $SLURM_TMPDIR
source $SLURM_TMPDIR/.venv/bin/activate

TODAY=$(TZ=":America/Montreal" date)
COMMIT_ID=$(git rev-parse --verify HEAD)
echo "Experiment $SLURM_JOB_ID ($PWD) start $TODAY on node $SLURMD_NODENAME (git commit id $COMMIT_ID)"
python -m torch.utils.collect_env

# Prepare data
DATASET_STORE='/home/sebwood/projects/def-franlp/sebwood/datasets'
datapath=$SLURM_TMPDIR/data
echo $datapath
mkdir $datapath
cp -r $DATASET_STORE/prepared_hf/imagenet1k.hfdatasets $datapath
ls $datapath

export HF_DATASETS_OFFLINE=1

SED_RES=$(sed -n "$SLURM_ARRAY_TASK_ID"p "$SOURCEDIR/experiments.dat")
eval "python main.py ${SED_RES} --datapath ${datapath}"
echo "Done"
