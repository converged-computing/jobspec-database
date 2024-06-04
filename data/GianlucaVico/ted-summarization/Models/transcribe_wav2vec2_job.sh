#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12000
#SBATCH --time=12:00:00
#SBATCH -A um_dke
#SBATCH --job-name="wav2vec2_complete"
#SBATCH --output="wav2vec2_log/transcribe_w2v2_mustc_%A.log"
#SBATCH --gres=gpu:pascal:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=my_email@email.com

module switch intel gcc/9
module load python/3.8.7
module load cuda/110
module load cudnn/8.0.5
module load cmake
module load LIBRARIES
module load intelmkl

export PATH=$HOME/.local/bin:$PATH
export KENLM_ROOT=$HOME/kenlm

cd $HOME/Documents/BSc-Thesis-AudioSumm/Models

DATA=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset
MUSTC=$WORK/MUST-C/en-cs/data
TEDDATA=$WORK/TED/Data
TED=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/TED
AMARADATA=$WORK/AMARA
AMARA=$WORK/AMARA
DATASET=$HOME/Documents/BSc-Thesis-AudioSumm/BuildDataset/integrated_data.csv

echo ted
echo $TED
### Execute your application
nvidia-smi
## python3 test_gpu.py
#CUDA_VISIBLE_DEVICES=""
python3 train_wav2vec2.py \
    -l $WORK/asr_logs \
    -o .. \
    -d cuda \
    -t $TED \
    --ted-data $TEDDATA \
    --amara $AMARA \
    --amara-data $AMARADATA \
    --must-c $MUSTC \
    --dataset $DATASET \
    --test-output .. \
    --transcribe \
    -m $HPCWORK/wav2vec2_mustc/checkpoint-15000-best
