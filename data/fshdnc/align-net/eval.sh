#!/bin/bash
#SBATCH -J eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_2002820
#SBATCH --output=/scratch/project_2002820/lihsin/align-lang/log-eval-%j.out
#SBATCH --error=/scratch/project_2002820/lihsin/align-lang/log-eval-%j.err

echo "START $SLURM_JOBID: $(date)"

function on_exit {
seff $SLURM_JOBID
gpuseff $SLURM_JOBID
echo "END $SLURM_JOBID: $(date)"
}
trap on_exit EXIT

#CKPT_PATH="/scratch/project_2002820/lihsin/align-lang/model_no_training.pt"
#CKPT_PATH="model_20210104-231744.pt" # trained for 7 epochs
CKPT_PATH="model_20210113-220042.pt" # on training set, for about 40 minutes

#POS_DICT="data/laser-test-set/dedup_src_trg_wmt-positives.json"
#SRC_SENT="data/laser-test-set/wmt-en.txt.dedup"
#TRG_SENT="data/laser-test-set/wmt-fi.txt.dedup"
POS_DICT="data/positives/dedup_src_trg_test-positives.json"
SRC_SENT="data/eng-fin/test.src.dedup"
TRG_SENT="data/eng-fin/test.trg.dedup"

echo -e "MODEL\t$CKPT_PATH\tDICT\t$POS_DICT\tSRC\t$SRC_SENT\tTRG\t$TRG_SENT"

module purge
module load pytorch/1.3.1
python3 evaluation.py \
    --src-sentences $SRC_SENT \
    --trg-sentences $TRG_SENT \
    --ckpt-path $CKPT_PATH \
    --positive-dict $POS_DICT

