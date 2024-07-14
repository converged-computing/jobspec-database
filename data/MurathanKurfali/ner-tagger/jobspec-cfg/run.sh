#!/bin/bash

usage="Usage: $0 [OPTIONS]
Fine-tune a transformer model for a specific task.

Options:
    --system-type        Type of the system to be used. (required)
    --model-name          Name of the BERT model to be fine-tuned. (required)
    --use-weighted-loss  Flag to activate weighted loss during training. Set to 'true' or '1' to enable. (default=False)
    --data-dir           Directory for data. (default=data)
    --run-count          Number of fine-tuning executions with different seeds. (default=4)
    -h                   Show this help message and exit.

Minimal usage: $0 --system-type <system-name> --model-name <target-model> --use-weighted-loss <loss_choice>
Example: $0 --system-type 'a' --model-name 'bert-base-cased' --use-weighted-loss 'true'
"

# Default values
data_dir="data"
run_count=4
system_type=""
model_name=""
use_weighted_loss="False"

# Display usage if no arguments provided
if [[ $# -eq 0 ]]; then
    printf "${usage}"
    exit 1
fi

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h) printf "$usage"
            exit 0
            ;;
        --system-type)
            system_type="$2"
            shift 2
            ;;
        --model-name)
            model_name="$2"
            shift 2
            ;;
        --use-weighted-loss)
            use_weighted_loss="$2"
            shift 2
            ;;
        --data-dir)
            data_dir="$2"
            shift 2
            ;;
        --run-count)
            run_count="$2"
            shift 2
            ;;
        *)
            echo "Invalid argument: $1"
            printf "${usage}"
            exit 1
            ;;
    esac
done

# Check for mandatory arguments
if [ -z "$system_type" ] || [ -z "$model_name" ]; then
    echo "Error: Missing required arguments."
    printf "${usage}"
    exit 1
fi


for (( i = 0; i < run_count; i++ )); do
  seed=$(( RANDOM % 100001 ))
  echo "Run number $((i+1)) of $run_count"
  echo "Seed: ${seed} ----"
  python -u scripts/run_ner.py \
        --model_name_or_path "${model_name}" \
        --task_name "ner" \
        --train_file "${data_dir}/system_${system_type}/train.json" \
        --validation_file "${data_dir}/system_${system_type}/validation.json" \
        --test_file "${data_dir}/system_${system_type}/test.json" \
        --output_dir "saved_models/system_${system_type}/${model_name}_${seed}" \
        --text_column_name "tokens" \
        --label_column_name "ner_tags" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 64 \
        --do_train \
        --use_weighted_loss "${use_weighted_loss}"\
        --seed ${seed} \
        --do_eval \
        --do_predict \
        --overwrite_cache \
        --overwrite_output_dir \
        --return_entity_level_metrics \
        --save_total_limit 2 \
        --greater_is_better "True" \
        --metric_for_best_model "eval_overall_f1" \
        --load_best_model_at_end \
        --save_strategy steps \
        --evaluation_strategy steps \
        --save_steps 1000 \
        --eval_steps 1000
done
