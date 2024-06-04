#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=02:30:00
#SBATCH --job-name=pred_doc
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --array=1-3

export PATH="$PATH:/home1/s3412768/.local/bin"

# Load modules
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0 

#load environment
source /home1/s3412768/.envs/nmt2/bin/activate


language=$1 # target language
model_type=$2 # type of experiment (baseline, genre_aware, genre_aware_token)
corpus=$3 # corpus to evaluate on
# use_tok=$3 # yes or no
# genre=$5 # genre to fine-tune on 

# corpus="MaCoCu"
train_corpus="MaCoCu"
exp_type="fine_tune" # type of model (e.g. fine_tuned or from_scratch.)

root_dir="/scratch/s3412768/genre_NMT/en-$language"

if [ $language = 'hr' ]; then
    model="Helsinki-NLP/opus-mt-en-sla"
elif 
    [ $language = 'tr' ]; then
    model="Helsinki-NLP/opus-mt-tc-big-en-tr"
else
    model="Helsinki-NLP/opus-mt-en-${language}"
fi


if [ $model_type = 'genre_aware' ] || [ $model_type = 'genre_aware_token' ]; then
    test_file="${root_dir}/data/${corpus}.en-$language.doc.test.tag.tsv"
    test_on="${corpus}.en-$language.doc.test.tag.tsv"
elif [ $model_type = 'baseline' ]; then
    test_file="${root_dir}/data/${corpus}.en-$language.doc.test.tsv"
    test_on="${corpus}.en-$language.doc.test.tsv"
else
    echo "Invalid model type"
    exit 1
fi

if [ $language == 'hr' ]; then 
    # add >>hrv<< in front of each line in the test file
    test_file_hr="${root_dir}/data/${test_on}.hrv"
    if [[ ! -f $test_file_hr ]]; then
        echo "Test file for hr not found, create it"
        awk '{print ">>hrv<< " $0}' $test_file > $test_file_hr
    fi
    test_file=$test_file_hr
fi


echo "test file: $test_file"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           


model_type="doc_${model_type}_${SLURM_ARRAY_TASK_ID}"

checkpoint=$root_dir/models/$exp_type/$model_type/$train_corpus/checkpoint-*

echo "Checkpoint: $checkpoint"


log_file="/scratch/s3412768/genre_NMT/en-$language/logs/$exp_type/$model_type/eval_${corpus}.log"
if [ ! -d "$root_dir/logs/$exp_type/$model_type" ]; then
    mkdir -p $root_dir/logs/$exp_type/$model_type
fi

echo "log file: $log_file"
echo "model type: $model_type"
echo "model: $model"

python /home1/s3412768/Genre-enabled-NMT/src/train.py \
    --root_dir $root_dir \
    --train_file $test_file \
    --dev_file $test_file \
    --test_file $test_file \
    --gradient_accumulation_steps 1 \
    --batch_size 32 \
    --gradient_checkpointing \
    --adafactor \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --learning_rate 1e-4 \
    --exp_type $exp_type \
    --model_type $model_type \
    --model_name $model \
    --predict \
    --eval \
    --checkpoint $checkpoint \
    &> $log_file





# eval_file=$test_on
# out_file="$(cut -d'.' -f1 <<<"$test_on")"

# out=$root_dir/eval/$exp_type/$model_type/${out_file}_predictions.txt

# # split file 

# python /home1/s3412768/Genre-enabled-NMT/src/split_docs_for_eval.py --input_file $out  --output_file $out.split

# # eval on the file wothout docs
# eval="$root_dir/data/${out_file}.en-$language.test.tsv"


# echo "Original Output file: $out"
# out=${out}.split
# echo "Split file for evaluation: $out.split"
# echo "Eval file: $eval"


# ref=${eval}.ref
# src=${eval}.src


# # deactivate the env used for predictions
# deactivate
# # remove the module used for predictions and load the new one
# module purge
# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
# source $HOME/.envs/nmt_eval/bin/activate
# set -eu -o pipefail


# # check if ref and src files exist and create them if not
# if [[ ! -f $ref ]]; then
#     echo "Reference file $ref not found, create it"
#     # First check if the file exists in the data folder
#     if [[ -f $eval ]]; then
#         # If so, extract the reference column
#         cut -d $'\t' -f2 $eval > "$ref"
#     else
#         echo "File $eval not found"
#     fi
# fi

# if [[ ! -f $src ]]; then
#     echo "Source file $src not found, create it"
#     # First check if the file exists in the data folder
#     if [[ -f $eval ]]; then
#         # If so, extract the source column
#         cut -d $'\t' -f1 $eval > "$src"
#     else
#         echo "File $eval not found"
#     fi
# fi


# if [[ ! -f $out ]]; then
#     echo "Output file $out not found, skip evaluation"
# else

#     # First put everything in 1 file
#     sacrebleu $out -i $ref -m bleu ter chrf --chrf-word-order 2 > "${out}.eval.sacre"
#     # Add chrf++ to the previous file
#     sacrebleu $out -i $ref -m chrf --chrf-word-order 2 >> "${out}.eval.sacre"
#     # Write only scores to individual files
#     sacrebleu $out -i $ref -m bleu -b > "${out}.eval.bleu"
#     sacrebleu $out -i $ref -m ter -b > "${out}.eval.ter"
#     sacrebleu $out -i $ref -m chrf -b > "${out}.eval.chrf"
#     sacrebleu $out -i $ref -m chrf --chrf-word-order 2 -b > "${out}.eval.chrfpp"

#     comet-score -s $src -t $out -r $ref > "${out}.eval.comet"

# fi