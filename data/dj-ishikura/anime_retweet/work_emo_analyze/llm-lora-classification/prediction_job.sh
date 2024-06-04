#PBS -q gLiotq
#PBS -l select=1:vnode=xind01:ncpus=8:mem=6G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.31.0
#PBS -k doe -j oe

cd ${PBS_O_WORKDIR}

TORCH_HOME=`pwd`/.cache/torch
TRANSFORMERS_CACHE=`pwd`/.cache/transformers
HF_HOME=`pwd`/.cache/huggingface
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME
export TORCH_USE_CUDA_DSA=1

INPUT_DIR=/work/n213304/learn/anime_retweet_2/extra_anime_tweet_text

OUTPUT_DIR=./prediction

mkdir -p $OUTPUT_DIR

for input_path in $INPUT_DIR/*.jsonl
do
    filename=$(basename "$input_path" .jsonl)
    output_path="${OUTPUT_DIR}/${filename}.json"
    # output_path="${OUTPUT_DIR}/tweet_data_randam_text.jsonl"
    # input_path="/work/n213304/learn/anime_retweet_2/tweet_data_randam_text.jsonl"
    if [ ! -f $output_path ]; then
        poetry run accelerate launch --mixed_precision=bf16 src/prediction_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 2 --lr 5e-05 --input_path ${input_path} --output_path ${output_path}
    fi
done

# input_path=/work/n213304/learn/anime_retweet_2/extra_anime_tweet_text/2020-01-7.jsonl
# output_path=/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/prediction/2020-01-7.json

# poetry run accelerate launch --mixed_precision=bf16 src/prediction_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 2 --lr 5e-05 --input_path ${input_path} --output_path ${output_path}
