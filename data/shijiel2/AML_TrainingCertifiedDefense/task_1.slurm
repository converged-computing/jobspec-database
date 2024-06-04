#!/bin/bash
#SBATCH --job-name=task_1
#SBATCH --output=task_1.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=50:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:4
#SBATCH -q gpgpudeeplearn
#SBATCH -A punim1629
#SBATCH --mail-user=404notfxxkingfound@gmail.com
#SBATCH --mail-type=ALL


echo "Loading required modules"
module load fosscuda/2020b
module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0

echo "Install libs" 
# pip3 install --upgrade pip
# pip3 install --user torch torchvision torchaudio
# pip3 install --user foolbox
# pip3 install --user git+https://github.com/fra31/auto-attack
# pip3 install --user scikit-learn
# pip3 install --user matplotlib

pip3 install statsmodels
# pip3 install -e .
pip3 install --user tqdm
pip3 install --user scikit-learn
pip3 install --user matplotlib

echo "Good to go!"
cd opacus/examples
python3 automator.py
# python3 -m torch.distributed.launch --nproc_per_node=4 automator.py