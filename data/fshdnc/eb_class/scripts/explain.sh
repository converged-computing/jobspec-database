#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=project_2002820
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=test
# --gres=gpu:v100:1
#SBATCH -e /scratch/project_2002820/lihsin/eb_class/output/%j.err
#SBATCH -o /scratch/project_2002820/lihsin/eb_class/output/%j.out

set -euo pipefail
echo "START: $(date)"

module purge
export SING_IMAGE=$(pwd)/eb_class_latest.sif
export TRANSFORMERS_CACHE=$(realpath cache)

#BERT_PATH=5832455/0_BERT
#cp -r $BERT_PATH $LOCAL_SCRATCH

echo "-------------SCRIPT--------------" >&2
cat $0 >&2
echo -e "\n\n\n" >&2

srun singularity exec --nv -B /scratch:/scratch $SING_IMAGE \
    python3 -m finnessayscore.explain\
    --batch_size 1 \
    --model_type whole_essay \
    --whole_essay_overlap 5 \
    --max_length 512 \
    --jsons data/ismi-kirjoitelmat-parsed.json
    #--epochs 20 \
    #--lr 2e-5 \
    #--grad_acc 1 \
    #
    #--use_label_smoothing \
    #--smoothing 0.0 \
    #
    #--bert_path $BERT_PATH


seff $SLURM_JOBID
echo "END: $(date)"
