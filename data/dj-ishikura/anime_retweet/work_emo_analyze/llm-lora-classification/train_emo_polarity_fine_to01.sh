#PBS -q gLiotq
#PBS -l select=1:ncpus=8:mem=64G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.31.0
#PBS -k doe -j oe

echo "a"

cd ${PBS_O_WORKDIR}

TORCH_HOME=`pwd`/.cache/torch
TRANSFORMERS_CACHE=`pwd`/.cache/transformers
HF_HOME=`pwd`/.cache/huggingface
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME
export TORCH_USE_CUDA_DSA=1

# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 1 --lr 5e-05

poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 2 --lr 5e-05

# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 2 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 2 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 2 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 3 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 3 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 3 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 4 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 4 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 4 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 2 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 2 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 2 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 3 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 3 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 3 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 4 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 4 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 64 --epochs 4 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 2 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 2 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 2 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 3 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 3 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 3 --lr 5e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 4 --lr 1e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 4 --lr 2e-05
# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 128 --epochs 4 --lr 5e-05

# poetry run accelerate launch --mixed_precision=bf16 src/train_emo_polarity.py --model_name rinna/japanese-gpt-neox-3.6b