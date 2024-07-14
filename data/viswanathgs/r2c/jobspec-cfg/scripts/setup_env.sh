#!/bin/bash

# Basically following steps in r2c/README.md

module load anaconda3/5.0.1
source deactivate

module purge
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1
module load FAISS/010818/gcc.5.4.0/anaconda3.5.0.1

conda create --clone fair_env_latest_py3 -n vcr
source activate vcr

# torchvision layers branch (for RoI Pooling)
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d

# allennlp
BASEDIR=$(dirname $0)/..
pip install -r "$BASEDIR"/allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0

python -m spacy download en_core_web_sm

pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# tensorflow (optional, only if training and generating BERT embeddings)
pip uninstall -y tensorflow tensorflow-gpu protobuf
pip install tensorflow tensorflow-gpu

echo "Setup conda env: vcr"
source deactivate
