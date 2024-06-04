#!/bin/bash
#SBATCH -J custom_dataset_train                 # Job name
#SBATCH --account=gts-rs275                     # charge account
#SBATCH -N 1 --gres=gpu:RTX_6000:2              # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=128G                       # Memory per core
#SBATCH -t 1440                                 # Duration of the job (Ex: 15 mins)
#SBATCH -q inferno                              # QOS Name
#SBATCH -o Report-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=cchen847@gatech.edu         # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3/2022.05                   # Load module dependencies
conda create -n bm ipython python=3.8 -y
conda activate bm
conda install pytorch==1.12.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -U -r requirements.txt
pip install -e .
python -m spacy download en_core_web_md
python analysis.py
