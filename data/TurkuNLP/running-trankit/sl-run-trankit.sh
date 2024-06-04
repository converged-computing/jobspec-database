#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -p gputest
#SBATCH -t 00:07:00
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2005092
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

mkdir -p logs

echo "START: $(date)"

echo "installing venv"

python3 -m venv venv-trankit
source venv-trankit/bin/activate
#export TRANSFORMERS_CACHE=cachedir_v
echo "venv done"
echo 
pip3 install --upgrade pip
pip3 install setuptools-rust
pip3 install trankit==1.1.0
pip3 install transformers


echo "Parsing"

cat $1 | srun python3 parse.py | gzip > output.conllu.gz
