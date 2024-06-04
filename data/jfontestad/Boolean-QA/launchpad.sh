#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"

module load anaconda
export TRANSFORMERS_CACHE=/scratch4/danielk/schaud31

#init virtual environment if needed
#conda remove -n toy_classification_env --all
#conda create -n toy_classification_env python=3.8

conda activate toy_classification_env # open the Python environment

#conda install torch pytorch torchvision torchaudio pytorch-cuda=11.7 faiss-gpu=1.7.3 cudatoolkit=11.3 -c pytorch -c nvidia
conda config --set allow_conda_downgrades true
conda install -c conda-forge faiss-gpu
conda install faiss-gpu

#clear cache
pip cache purge

pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install petals
conda config --set allow_conda_downgrades false
#conda list

#clear cache
#pip install huggingface_hub["cli"]
#huggingface-cli delete-cache

#runs your code
#srun python classification.py  --experiment "overfit" --small_subset False --device cuda --model "bert-base-uncased" --batch_size "32" --lr 1e-4 --num_epochs 9
#srun python rag_classification.py  --experiment "rag" --small_subset True --device cuda --model "facebook/rag-token-nq" --batch_size "1" --lr 1e-4 --num_epochs 3
#srun python bloomz.py  --experiment "bloomz" --model "bigscience/bloomz-560m"
srun python petals_bloomz.py --experiment "petals_finetuned" --model "bigscience/bloomz-560m" --small_subset True --batch_size "1" --lr 1e-4 --num_epochs 3
