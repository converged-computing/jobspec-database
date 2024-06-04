#!/bin/bash
#SBATCH --job-name=WGAN
#SBATCH --partition=dgx_normal_q
#SBATCH --time=46:00:00
#SBATCH -A HPCBIGDATA2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --propagate=STACK
export MASTER_PORT=8888
#export WORLD_SIZE=4
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
# echo "${SLURM_NODELIST:7:1}"
# echo "${SLURM_NODELIST:8:3}"
# echo "MASTERs_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
##only for tinkercliffs
if [ ${SLURM_NODELIST:6:1} == "[" ]; then
    echo "MASTER_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
module reset
    export MASTER_ADDR=${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:3}
else
    echo "MASTER_ADDR="${SLURM_NODELIST}
    export MASTER_ADDR=${SLURM_NODELIST}
fi
mkdir -p $SLURM_JOBID
export weight_path="./$SLURM_JOBID/"
module reset
module restore cu117
module list
module load containers/apptainer
source ~/.bashrc
conda activate test
export imagefile="/projects/synergy_lab/ayush/containers/pytorch_23.03.sif"

export BASE="apptainer  exec --nv --writable-tmpfs --bind=/projects/synergy_lab,/cm/shared,${TMPFS} ${imagefile} "

export conda_env="test"
#export CMD="conda run -n ${conda_env} python main.py"

export CMD="python3 main.py --save_path $weight_path"
echo "running command with srun: $BASE $CMD"
srun  --unbuffered --wait=120 --kill-on-bad-exit=0 --cpu-bind=none $CMD