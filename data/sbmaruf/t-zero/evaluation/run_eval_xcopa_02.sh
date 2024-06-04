#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PA100q
#SBATCH -n 1
#SBATCH --nodelist=node03
#SSSSSSBATCH --gres=gpu:1

module load cuda11.6/toolkit/11.6.0
source ~/anaconda3/bin/activate py3

export CUDA_VISIBLE_DEVICES=2



mkdir -p dumped
DATASET_NAME="xcopa"
for MODEL_SIGNATURE in "tr13f-6b3-ml-t0-lmtoks341b-t0toks4b2-xp3capmixnewcodelonglossseq" "tr13f-6b3-ml-t0-lmtoks341b-t0toks4b2-xp3mt" ; do
    for DATASET_CONFIG_NAME in "et" "ht" "it" "id" "qu" "sw" "zh" "ta" "th" "tr" "vi" ; do
        MODEL_NAME_OR_PATH="bigscience/$MODEL_SIGNATURE"
        for TEMPLATE_NAME in "exercise" "i_am_hesitating" "plausible_alternatives" "C1 or C2? premise, so/because…" "best_option" "more likely" "cause_effect" "choose" ; 
            do
            TEMPLATE_CONFIG_NAME=$DATASET_NAME/$DATASET_CONFIG_NAME
            TEMPLATE_CONFIG_FLAG=${TEMPLATE_CONFIG_NAME//[\/ ]/_}
            TEMPLATE_FLAG=${TEMPLATE_NAME//[\/ ]/_}
            OUTPUT_DIR="dumped_trans"/$MODEL_SIGNATURE/$DATASET_NAME"_"$DATASET_CONFIG_NAME"_"$TEMPLATE_CONFIG_FLAG"_"$TEMPLATE_FLAG
	        if [ ! -d $OUTPUT_DIR ] 
                then
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
            TEMPLATE_CONFIG_NAME=$DATASET_NAME/"en"
            TEMPLATE_CONFIG_FLAG=${TEMPLATE_CONFIG_NAME//[\/ ]/_}
            TEMPLATE_FLAG=${TEMPLATE_NAME//[\/ ]/_}
            OUTPUT_DIR="dumped_trans"/$MODEL_SIGNATURE/$DATASET_NAME"_"$DATASET_CONFIG_NAME"_"$TEMPLATE_CONFIG_FLAG"_"$TEMPLATE_FLAG
	        if [ ! -d $OUTPUT_DIR ] 
                then
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


mkdir -p dumped
DATASET_NAME="super_glue"
for MODEL_SIGNATURE in "tr13f-6b3-ml-t0-lmtoks341b-t0toks4b2-xp3capmixnewcodelonglossseq" "tr13f-6b3-ml-t0-lmtoks341b-t0toks4b2-xp3mt" ; do
    for DATASET_CONFIG_NAME in "copa" ; do
        MODEL_NAME_OR_PATH="bigscience/$MODEL_SIGNATURE"
        for TEMPLATE_NAME in "exercise" "i_am_hesitating" "plausible_alternatives" "C1 or C2? premise, so/because…" "best_option" "more likely" "cause_effect" "choose" ; 
            do
            TEMPLATE_CONFIG_NAME=$DATASET_NAME/$DATASET_CONFIG_NAME
            TEMPLATE_CONFIG_FLAG=${TEMPLATE_CONFIG_NAME//[\/ ]/_}
            TEMPLATE_FLAG=${TEMPLATE_NAME//[\/ ]/_}
            OUTPUT_DIR="dumped_trans"/$MODEL_SIGNATURE/$DATASET_NAME"_"$DATASET_CONFIG_NAME"_"$TEMPLATE_CONFIG_FLAG"_"$TEMPLATE_FLAG
	        if [ ! -d $OUTPUT_DIR ] 
                then
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



