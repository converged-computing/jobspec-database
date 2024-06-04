#!/bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=16:ngpus=1:mem=110gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -P 13003565
##PBS -N convert_only.log
##PBS -o convert_only.log


################################################# 
# # echo PBS: qsub is running on $PBS_O_HOST
# echo PBS: executing queue is $PBS_QUEUE
# echo -e "Work folder is $PWD\n\n"
#
# echo PBS: working directory is $PBS_O_WORKDIR
# echo PBS: job identifier is $PBS_JOBID
# echo PBS: job name is $PBS_JOBNAME
# echo PBS: node file is $PBS_NODEFILE
# echo PBS: current home directory is $PBS_O_HOME
# echo PBS: PATH = $PBS_O_PATH
# #################################################
# cd $PBS_O_WORKDIR
# echo -e "Work folder is $PWD\n\n"
#
# #################################################
# source /data/projects/13003565/geyu/miniconda3/etc/profile.d/conda.sh
# conda activate seaeval
# echo "Virtual environment activated"
#
# #################################################
#################################################



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

echo "MODEL_INDEX=$MODEL_INDEX"
echo "LISTEN_FOLDER=$LISTEN_FOLDER"


if [[ -e "$LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32/pytorch_model.bin" ]]; then
  echo "Model bin exists."
else
  echo "Model bin not exists" 
  mkdir -p $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
  cp -r ./helper_configs/* $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
  python convert_lora.py $LISTEN_FOLDER/$MODEL_INDEX $BASE_MODEL $LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32/
  echo "Created a normal checkpoints"
fi

# mkdir -p converted_checkpoint/$MODEL_INDEX-fp32
# cp -r converted_checkpoint/helper_configs/* converted_checkpoint/$MODEL_INDEX-fp32/
# python zero_to_fp32.py $LISTEN_FOLDER/$MODEL_INDEX $LISTEN_FOLDER/$MODEL_INDEX-fp32/pytorch_model.bin

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

#mkdir -p /data/projects/13003558/pretrain_output_results/$MODEL_INDEX-results

MODEL_PATH=$LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-fp32
echo $MODEL_PATH
##### 
MODEL_NAME=$MODEL_PATH
GPU=0
BZ=8
#####

# for EVAL_MODE in public_test_few_shot hidden_test_few_shot
# do

#     mkdir -p converted_checkpoint/$MODEL_NAME-results/$EVAL_MODE

#     for ((i=1; i<=1; i++))
#     do
#         bash eval.sh cross_mmlu $MODEL_NAME $GPU $BZ $i $EVAL_MODE              2>&1 | tee converted_checkpoint/$MODEL_NAME-results/$EVAL_MODE/cross_mmlu_p$i.log
#         bash eval.sh cross_logiqa $MODEL_NAME $GPU $BZ $i $EVAL_MODE            2>&1 | tee converted_checkpoint/$MODEL_NAME-results/$EVAL_MODE/cross_logiqa_p$i.log
#     done
# done


echo "MODEL_NAME=$MODEL_NAME"

for EVAL_MODE in hidden_test
do
    TARGET_DIR=$LISTEN_FOLDER/converted_checkpoint/$MODEL_INDEX-results/$EVAL_MODE
    mkdir -p $TARGET_DIR/log

    for ((i=1; i<=1; i++))
    do
	      bash scripts/eval.sh cross_xquad $MODEL_NAME $GPU $BZ $i $EVAL_MODE $TARGET_DIR        2>&1 | tee $TARGET_DIR/log/cross_xquad_p$i.log
        bash scripts/eval.sh cross_mmlu $MODEL_NAME $GPU $BZ $i $EVAL_MODE $TARGET_DIR         2>&1 | tee $TARGET_DIR/log/cross_mmlu_p$i.log
        bash scripts/eval.sh cross_logiqa $MODEL_NAME $GPU $BZ $i $EVAL_MODE $TARGET_DIR           2>&1 | tee $TARGET_DIR/log/cross_logiqa_p$i.log
    done
done


# rm -rf $MODEL_PATH
rm -rf $LISTEN_FOLDER/$MODEL_INDEX
# echo "$MODEL_PATH CLEANED"




#cp -r converted_checkpoint/$MODEL_INDEX-results/* /data/projects/13003558/pretrain_output_results/$MODEL_INDEX-results/

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




