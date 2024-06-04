#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PA100q
#SBATCH -n 1
#SBATCH --nodelist=node03
#SSSSSSBATCH --gres=gpu:1

module load cuda11.6/toolkit/11.6.0
source ~/anaconda3/bin/activate py3

export CUDA_VISIBLE_DEVICES=6



DATASET_NAME="paws-x"
TEMPLATE_CONFIG_NAME="paws-x/en"
for MODEL_SIGNATURE in "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-xp3capmix" "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-p31"; do
    DATASET_CONFIG_NAME="en"
    MODEL_NAME_OR_PATH="bigscience/$MODEL_SIGNATURE"
    for TEMPLATE_NAME in "task_description-no-label" "Meaning" "context-question-no-label" "Rewrite-no-label" "context-question" "Concatenation" "Concatenation-no-label" "Meaning-no-label" "PAWS-ANLI GPT3" "Rewrite" "PAWS-ANLI GPT3-no-label" ; 
        do
	    TEMPLATE_CONFIG_FLAG=${TEMPLATE_CONFIG_NAME//[\/ ]/_}
	    TEMPLATE_FLAG=${TEMPLATE_NAME//[\/ ]/_}
        OUTPUT_DIR="dumped"/$MODEL_SIGNATURE/$DATASET_NAME"_"$DATASET_CONFIG_NAME"_"$TEMPLATE_CONFIG_FLAG"_"$TEMPLATE_FLAG
        sleep 2
        if [ ! -d $OUTPUT_DIR ] 
            then
            sleep 2
            mkdir -p $OUTPUT_DIR
            python run_eval.py \
            --dataset_name $DATASET_NAME \
            --dataset_config_name $DATASET_CONFIG_NAME \
	        --dataset_split "test" \
            --template_config_name $TEMPLATE_CONFIG_NAME \
            --template_name "$TEMPLATE_NAME" \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --output_dir $OUTPUT_DIR
        fi
    done
done


for MODEL_SIGNATURE in "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-xp3capmix" "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-p31"; do
    for DATASET_CONFIG_NAME in "de" "en" "es" "fr" "ja" "ko" "zh"; do
        MODEL_NAME_OR_PATH="bigscience/$MODEL_SIGNATURE"
        for TEMPLATE_NAME in "task_description-no-label" "Meaning" "context-question-no-label" "Rewrite-no-label" "context-question" "Concatenation" "Concatenation-no-label" "Meaning-no-label" "PAWS-ANLI GPT3" "Rewrite" "PAWS-ANLI GPT3-no-label" ; 
            do
            TEMPLATE_CONFIG_FLAG=${TEMPLATE_CONFIG_NAME//[\/ ]/_}
            TEMPLATE_FLAG=${TEMPLATE_NAME//[\/ ]/_}
            OUTPUT_DIR="dumped"/$MODEL_SIGNATURE/$DATASET_NAME"_"$DATASET_CONFIG_NAME"_"$TEMPLATE_CONFIG_FLAG"_"$TEMPLATE_FLAG
            sleep 2
	        if [ ! -d $OUTPUT_DIR ] 
                then
                sleep 2
                mkdir -p $OUTPUT_DIR
                python run_eval.py \
                --dataset_name $DATASET_NAME \
                --dataset_config_name $DATASET_CONFIG_NAME \
		        --dataset_split "test" \
                --template_config_name $TEMPLATE_CONFIG_NAME \
                --template_name "$TEMPLATE_NAME" \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --output_dir $OUTPUT_DIR
            fi
        done
    done
done


sbatch run_eval_xcopa_06.sh