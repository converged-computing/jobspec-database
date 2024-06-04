#PBS -q gLiotq
#PBS -l select=1:ncpus=8:mem=64G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.31.0
#PBS -k doe -j oe

echo "a"

output_parent_dir="/work/n213304/learn/anime_retweet_2/work_emo_analyze/llm-lora-classification/outputs/rinna__japanese-gpt-neox-3.6b"
batch_size=32

cd ${PBS_O_WORKDIR}

TORCH_HOME=`pwd`/.cache/torch
TRANSFORMERS_CACHE=`pwd`/.cache/transformers
HF_HOME=`pwd`/.cache/huggingface
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME
export TORCH_USE_CUDA_DSA=1

# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity.py --model_name rinna/japanese-gpt-neox-3.6b