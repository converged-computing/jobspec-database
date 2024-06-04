#!/bin/bash
#SBATCH --account=Project_2000539
#SBATCH --time=24:00:00
##SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --cpus-per-task=4

#module load python-data/3.7.6-1
#source /projappl/project_2000539/faiss_distractors/venv-faissgpu/bin/activate

module load pytorch/1.11

#export TMPDIR=$LOCAL_SCRATCH
export PYTORCH_PRETRAINED_BERT_CACHE=$TMPDIR

PART=$1
PARTS=$2
DATAIN=$3 ## some .gz with sentences
DATAOUT=$4 ## where to store the embedded vectors

#zcat $DATAIN | python3 embed.py --thisjob $PART --jobs $PARTS --bert-model TurkuNLP/bert-base-finnish-cased-v1 --out $DATAOUT

python3 embed_sbert.py --thisjob $PART --jobs $PARTS --in-file $DATAIN --bert-tokenizer sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --sbert-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --out $DATAOUT

