#!/bin/bash
#SBATCH --job-name="finetuning"
#SBATCH -D /gpfs/scratch/bsc88/bsc88080/stance_models/
#SBATCH --output=logs/finetuning_%j.out
#SBATCH --error=logs/finetuning_%j.err
#SBATCH --nodes=1
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task=128
#SBATCH --time=2-0:00:00

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

#MODULE LOADING

if uname -a | grep -q amd
then
        module load cmake/3.18.2 gcc/10.2.0 rocm/5.1.1 mkl/2018.4 intel/2018.4 python/3.7.4
        source /gpfs/projects/bsc88/projects/catalan_evaluation/venv/bin/activate
        export LD_LIBRARY_PATH=/gpfs/projects/bsc88/external-lib:$LD_LIBRARY_PATH
else
        module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 \
        atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 \
        python/3.7.4_ML arrow/3.0.0 text-mining/2.0.0 torch/1.9.0a0 torchvision/0.11.0
fi

SEED=1

#MODEL ARGUMENTS
SPLIT_STRATEGY=$1
MODEL="/gpfs/projects/bsc88/huggingface/models/"$2
DATASET_PATH="/gpfs/scratch/bsc88/bsc88080/stance_models/data/$SPLIT_STRATEGY/DynamicStance.py"
echo $DATASET_PATH
LEARN_RATE=0.00001
MODEL_NAME=$( basename $MODEL )
NUM_EPOCHS=10
BATCH_SIZE=8
MAX_LENGTH=512
GRADIENT_ACC_STEPS=1
BATCH_SIZE_PER_GPU=$(( $BATCH_SIZE*$GRADIENT_ACC_STEPS ))
DATASET_NAME=$( basename $DATASET_PATH )

TASK="d_stance"
SCRIPT="/gpfs/scratch/bsc88/bsc88080/stance_models/run_glue.py"


MODEL_ARGS=""
if test ! -z $TASK; then
        MODEL_ARGS="--task_name $TASK "
fi
MODEL_ARGS+=" \
 --model_name_or_path $MODEL \
 --dataset_name $DATASET_PATH \
 --do_train \
 --do_eval \
 --do_predict \
 --num_train_epochs $NUM_EPOCHS \
 --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
 --per_device_train_batch_size $BATCH_SIZE \
 --learning_rate $LEARN_RATE \
 --max_seq_length $MAX_LENGTH \
 --evaluation_strategy epoch \
 --save_strategy epoch \
 "



#OUTPUT ARGUMENTS

OUTPUT_DIR='output/'$MODEL_NAME'/'$SPLIT_STRATEGY
LOGGING_DIR='tb/'$MODEL_NAME'/'$SPLIT_STRATEGY
DIR_NAME=${DATASET_NAME}_${BATCH_SIZE_PER_GPU}_${LEARN_RATE}_${NUM_EPOCHS}_date_${DATETIME} #$DATETIME
CACHE_DIR='cache/'$OUTPUT_DIR/$DIR_NAME

OUTPUT_ARGS=" \
 --output_dir $OUTPUT_DIR/$DIR_NAME \
 --logging_dir $LOGGING_DIR/$DIR_NAME \
 --logging_strategy epoch \
 --cache_dir $CACHE_DIR \
 --load_best_model_at_end \
 "


export HF_HOME=$CACHE_DIR/huggingface
echo $SCRIPT
echo $DATASET_PATH

python $SCRIPT --seed $SEED $MODEL_ARGS $OUTPUT_ARGS
