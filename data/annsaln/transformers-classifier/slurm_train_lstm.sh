#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -t 1:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002026
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "START: $(date)"

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load tensorflow/2.2
#source transformers3.4tf2.4/bin/activate

#export PYTHONPATH=/scratch/project_2002026/multilabel_bert/svregisters/lstm/transformer-classifier/transformers3.4tf3.4/lib/python3.7/site-packages:$PYTHONPATH

MODEL=$1
MODEL_ALIAS=$2
SRC=$3
TRG=$4
LR_=$5
EPOCHS_=$6
i=$7
BS=128

echo "MODEL:$MODEL"
echo "MODEL_ALIAS:$MODEL_ALIAS"
echo "SRC:$SRC"
echo "TRG:$TRG"
echo "LR:$LR_"
echo "EPOCHS:$EPOCHS_"
echo "i:$i"

export TRAIN_DIR=junkdata/$SRC
export DEV_DIR=junkdata/$TRG
export OUTPUT_DIR=models

mkdir -p "$OUTPUT_DIR"

for EPOCHS in $EPOCHS_; do
for LR in $LR_; do
for j in $i; do
echo "Settings: src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS"
echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/experiments.log
srun python train_lstm.py \
  --model_name $MODEL \
  --train $TRAIN_DIR/train.json \
  --dev $DEV_DIR/dev.json \
  --test $DEV_DIR/test.json \
  --input_format json \
  --lr $LR \
  --seq_len 512 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --threshold 0.5 \
  --max_lines 50 \
  --log_file logs/train_$MODEL_ALIAS-$SRC-$TRG.tsv \
  --test_log_file logs/test_$MODEL_ALIAS-$SRC-$TRG.tsv \
  --load_model $OUTPUT_DIR/xmlr-de-en-es-fi-fr-se-de-en-es-fi-fr-se-lr2e-6-ep6-1.h5 \
#  --output_file $OUTPUT_DIR/$MODEL_ALIAS-$SRC-$TRG-lr$LR-ep$EPOCHS-$j.h5 \
#  --multiclass
#--output_file $OUTPUT_DIR/model.h5 \
#  --load_model $OUTPUT_DIR/model_nblocks3-ep10-2.h5 \
#--load_model $OUTPUT_DIR/model.h5 \
# bigger model xmlr-sv-sv-lr7e-6-ep10-1.h5 f-score 80
# smaller xmlr-sv-test-sv-test-lr2e-05-ep5-1.h5 f-score 87
done
done
done

echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/completed.log


#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"
