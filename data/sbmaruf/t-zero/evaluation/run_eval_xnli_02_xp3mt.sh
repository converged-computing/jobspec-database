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
DATASET_NAME="xnli"
for MODEL_SIGNATURE in "tr13f-6b3-ml-t0-lmtoks341b-t0toks4b2-xp3capmixnewcodelonglossseq" "tr13f-6b3-ml-t0-lmtoks341b-t0toks4b2-xp3mt" "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-xp3capmix" "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-p31" "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-p31lossseq" "tr13f-6b3-ml-t0-lmtoks341b-t0toks13b-xp3capmixlossseq"; do
    for DATASET_CONFIG_NAME in "en" "fr" "es" "de" "el" "bg" "ru" "tr" "ar" "vi" "th" "zh" "hi" "sw" "ur"; do
        MODEL_NAME_OR_PATH="bigscience/$MODEL_SIGNATURE"
        for TEMPLATE_NAME in "take the following as truth" "does this imply" "GPT-3 style" "does it follow that" "based on the previous passage" "guaranteed true" "should assume" "must be true" "can we infer" "justified in saying" "claim true/false/inconclusive" "consider always/sometimes/never" "always/sometimes/never" "guaranteed/possible/impossible" "MNLI crowdsource" ; 
            do
            TEMPLATE_CONFIG_NAME=$DATASET_NAME/$DATASET_CONFIG_NAME
            TEMPLATE_CONFIG_FLAG=${TEMPLATE_CONFIG_NAME//[\/ ]/_}
            TEMPLATE_FLAG=${TEMPLATE_NAME//[\/ ]/_}
            OUTPUT_DIR="dumped_trans"/$MODEL_SIGNATURE/$DATASET_NAME"_"$DATASET_CONFIG_NAME"_"$TEMPLATE_CONFIG_FLAG"_"$TEMPLATE_FLAG
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

            TEMPLATE_CONFIG_NAME=$DATASET_NAME/"en"
            TEMPLATE_CONFIG_FLAG=${TEMPLATE_CONFIG_NAME//[\/ ]/_}
            TEMPLATE_FLAG=${TEMPLATE_NAME//[\/ ]/_}
            OUTPUT_DIR="dumped_trans"/$MODEL_SIGNATURE/$DATASET_NAME"_"$DATASET_CONFIG_NAME"_"$TEMPLATE_CONFIG_FLAG"_"$TEMPLATE_FLAG
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

python ~/run.py --nodes 03 --run --export 02
# sbatch run_eval_pawsx_00.sh
