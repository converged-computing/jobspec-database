#!/bin/bash
#SBATCH --gres=gpu:2         # Number of GPUs (per node)
#SBATCH --cpus-per-task=8    # Number of CPUs
#SBATCH --mem=48G            # memory (per node)
#SBATCH --time=00-48:00:00   # time (DD-HH:MM:SS)
#SBATCH --partition=main     # priority: unkillable > main > long
#SBATCH --job-name=Grokking  #

module load cuda/10.1
source ../grokking/bin/activate

filename=train.sh

chmod +x $filename
#cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 

. $filename

# ############## README : Before runing this file on the cluster #################
# #### see https://gist.github.com/Tikquuss/d412b678f0c116a0061016b19645294d#file-mila_cluster-sh ####

# module load python/3.7
# virtualenv grokking
# source grokking/bin/activate
# pip install --upgrade pip
# #https://pytorch.org/
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# git clone https://github.com/Tikquuss/lm_grokking
# cd lm_grokking
# pip install -r requirements.txt
# ### for `import pytorch_lightning as pl` issues
# pip3 install packaging
# pip install importlib-metadata
# pip install transformers -U
# ### for `from language_modelling import LMLightningModule` issues
# pip3 install python-dateutil
# pip uninstall attr
# pip install attrs

# tmux

# salloc --gres=gpu:2 -c 8 --mem=48Gb --time=48:00:00 --partition=main --job-name=Grokking
# chmod +x cluster.sh
# . cluster.sh

