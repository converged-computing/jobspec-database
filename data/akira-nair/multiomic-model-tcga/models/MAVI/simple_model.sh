#!/bin/sh
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o model_tf%j.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,BEGIN
#SBATCH --mail-user=akira_nair@brown.edu
# conda --version
# conda info --envs
# conda deactivate
# conda activate /users/anair27/anaconda/akira_conda
module load python/3.11.0 openssl/3.0.0 cuda/11.7.1 cudnn/8.2.0
source /users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/tf_gpu.venv/bin/activate
# pip install tensorflow
which python3
which python
nvidia-smi
python3 -c "import tensorflow as tf;tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)"
cd /users/anair27/data/anair27/singh-lab-TCGA-project/multiomic-model-tcga/
python3 models/MAVI/simple_model.py