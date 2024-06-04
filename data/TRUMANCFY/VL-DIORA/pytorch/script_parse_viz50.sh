#!/bin/bash
# sb --gres=gpu:titan_xp:rtx --cpus-per-task=16 --mem=100G coco_run.sh

export MASTER_ADDR="0.0.0.0"
export MASTER_PORT="8088"
export NODE_RANK=0

#SBATCH  --mail-type=ALL                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --gres=gpu:geforce_rtx_3090:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_IdD
#
# binary to execute
set -o errexit
source /itet-stor/fencai/net_scratch/anaconda3/bin/activate diora
export PYTHONPATH=/itet-stor/fencai/net_scratch/diora/pytorch/:$PYTHONPATH

# ebdde512
# e2bd512b

# bag: 30cb2364

# bed: a4df82a5

# table: 49d65919

# table: model not frozen 6cce0009

# chair not frozen 4d9e6436

# chair frozen: bb6f3ac1: model.step_50800.pt

### new loss

# table
# frozen: e7ed6d25
# not frozen: 0fe669fa

# c3c9f330

# 61d9004a: checkpoint 14300
# 7c2531b3; model 16800

# kl + freecls: 4d391393

# correct the number of classes: 472643c9 model.step_14300.pt

# use the classfication result: 0f271581 model.step_3100.pt

# a0d66393 12400

# b6d3adf7

# ecb61fe6 model.step_12400.pt

# 8de11564: model.


srun python diora/scripts/parse_viz.py \
    --batch_size 1 \
    --data_type viz \
    --emb resnet50 \
    --load_model_path ../log/569a1213/model.step_1000.pt \
    --model_flags ../log/569a1213/flags.json \
    --validation_path ./data/partit_data/3.bag/test \
    --validation_filter_length 20 \
    --word2idx './data/partit_data/partnet.dict.pkl' \
    --k_neg 5 \
    --freeze_model 1 \
    --cuda \
    --vision_type "bag"

echo finished at: `date`
exit 0;