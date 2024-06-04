#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --output=JO-%x.%j.out
#SBATCH --error=JO-%x.%j.err
#IGNBATCH --mem-per-gpu=16G

# ./bash_utils/env_setup.sh
module load StdEnv/2020  gcc/9.3.0  cuda/11.4 arrow/8.0.0 python/3.10
virtualenv --no-download "$ENV_PATH"
#virtualenv $ENV_PATH
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
#pip install --upgrade pip
pip install --no-index numpy==1.23.5
pip install --no-index -r requirements.txt
#pip install -r requirements.txt
pip install --no-deps /home/amirresm/files/research/summarization/adapters-0.1.2-py3-none-any.whl

./bash_utils/capability_check.sh

base_config_title=""
lang="java"
config_title="t5_base_compacter_${lang}_b16"

model="codet5-base"
adapter_config="compacter"

# Paths
prog_root=/home/amirresm/files/research/summarization
script_path=$prog_root/run_summarization_pure.py
bleu_path=$prog_root/bleu/bleu.py
rouge_path=$prog_root/rouge/rouge.py

storage_root=/home/amirresm/projects/def-fard/amirresm
parent_path=/home/amirresm/projects/def-fard/amirresm/outputs/rr_experiments/$lang/$config_title

data=$storage_root/data/CodeSearchNet/$lang
model_path=$storage_root/models/$model
# model_path=$storage_root/outputs/t5_experiments/$base_config_title

output_dir=$parent_path
logging_dir=$output_dir/logs

adapter_path=$output_dir/${config_title}_adapter
tokenizer_name_or_path=$output_dir/${config_title}_tokenizer
generation_output_path=$output_dir/gen_output

# Behaviors
do_train=1
do_eval=1
do_predict=1

source_prefix="summarize: "

text_column="code_tokens"
summary_column="docstring_tokens"
text_tokenized=1
summary_tokenized=1

train_adapter=1
preload_adapter=1

#Hyperparameters
learning_rate="5e-5"
weight_decay="0.01"
num_train_epochs="3.0"
warmup_steps=500
per_device_train_batch_size=16
per_device_eval_batch_size=16
max_source_length=512
max_target_length=256
generation_max_length=$max_target_length

# Configs
use_fast_tokenizer=1
train_tokenizer=0
pad_to_max_length=1
ignore_pad_token_for_loss=1
overwrite_output_dir=0

# Others
train_file="${data}/train.jsonl"
eval_file="${data}/valid.jsonl"
test_file="${data}/test.jsonl"

export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

${prog_root}/checkmem.sh 600 "${output_dir}/memuse.txt" &

echo "Bash going to Python..."
python3 $script_path \
    --config_title $config_title \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --do_train $do_train \
    --do_eval $do_eval \
    --do_predict $do_predict \
    --train_file ${train_file} \
    --validation_file ${eval_file} \
    --test_file ${test_file} \
    --text_column $text_column \
    --summary_column $summary_column \
	--text_tokenized $text_tokenized \
	--summary_tokenized $summary_tokenized \
    --source_prefix $source_prefix \
    --output_dir $output_dir \
    --overwrite_output_dir $overwrite_output_dir \
    --use_fast_tokenizer $use_fast_tokenizer \
    --train_tokenizer $train_tokenizer \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_train_epochs $num_train_epochs \
    --warmup_steps $warmup_steps \
    --predict_with_generate \
    --evaluation_strategy steps \
	--eval_steps "0.1" \
    --logging_strategy steps \
	--logging_steps "0.1" \
	--logging_dir $logging_dir \
    --save_total_limit 10 \
    --metric_path $bleu_path \
    --metric_path_alt $rouge_path \
    --train_adapter $train_adapter \
    --adapter_config $adapter_config \
    --adapter_path $adapter_path \
    --preload_adapter $preload_adapter \
    --generation_output_path $generation_output_path \
    --max_source_length $max_source_length \
    --max_target_length $max_target_length \
    --generation_max_length $generation_max_length \
    --pad_to_max_length $pad_to_max_length \
    --ignore_pad_token_for_loss $ignore_pad_token_for_loss \
    --max_train_samples 9999999999 \
    --max_eval_samples 9999999999 \
    --max_predict_samples 9999999999 \
    2>&1| tee "$output_dir/job_report.log"

