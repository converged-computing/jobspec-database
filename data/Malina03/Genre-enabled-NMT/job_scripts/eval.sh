#!/bin/bash
# Job scheduling info, only for us specifically
#SBATCH --time=01:00:00
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=20G

module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
source $HOME/.envs/nmt_eval/bin/activate
set -eu -o pipefail

# Calculate all metrics between two files
out_file=$1 # File produced by model ("{out_file}_predictions.txt")
lang=$2 # Target language
exp_type=$3 # type of experiment (fine_tuned or from_scratch.)
model_type=$4 # type of model (genre_aware, genre_aware_token -genres are added as proper tokens- or baseline)
genre=$5 # the genre that the model was trained on
eval_file=$6 # File to evaluate against


root_dir="/scratch/s3412768/genre_NMT/en-${lang}"
out=$root_dir/eval/$exp_type/$model_type/$genre/${out_file}_predictions.txt
eval=$root_dir/data/$eval_file

ref=$eval.ref
src=$eval.src

# check if ref and src files exist and create them if not
if [[ ! -f $ref ]]; then
    echo "Reference file $ref not found, create it"
    # First check if the file exists in the data folder
    if [[ -f $eval ]]; then
        # If so, extract the reference column
        cut -f2 $eval > $ref
    else
        echo "File $eval not found"
    fi
fi

if [[ ! -f $src ]]; then
    echo "Source file $src not found, create it"
    # First check if the file exists in the data folder
    if [[ -f $eval ]]; then
        # If so, extract the source column
        cut -f1 $eval > $src
    else
        echo "File $eval not found"
    fi
fi


if [[ ! -f $out ]]; then
	echo "Output file $out not found, skip evaluation"
else
	# NOTE: automatically get target language by last 2 chars of ref file
	# So assume it is called something like wiki.en-mt for example
	# Otherwise just manually specify it below
	
	# Skip whole BLEU/chrf section if last file already exists unless $force is set
	# if [[ -f "${out}.eval.chrfpp" ]]; then
	# 	echo "Eval file already exists, skip BLEU and friends"
	# else
	# First put everything in 1 file
	sacrebleu $out -i $ref -m bleu ter chrf --chrf-word-order 2 > ${out}.eval.sacre
	# Add chrf++ to the previous file
	sacrebleu $out -i $ref -m chrf --chrf-word-order 2 >> ${out}.eval.sacre
	# Write only scores to individual files
	sacrebleu $out -i $ref -m bleu -b > ${out}.eval.bleu
	sacrebleu $out -i $ref -m ter -b > ${out}.eval.ter
	sacrebleu $out -i $ref -m chrf -b > ${out}.eval.chrf
	sacrebleu $out -i $ref -m chrf --chrf-word-order 2 -b > ${out}.eval.chrfpp
	# fi	

	# Calculate BLEURT (pretty slow)
	# If error: 
	# module load cuDNN
	# module load GLibmm
	# if [[ -f "${out}.eval.bleurt" ]]; then
	# 	echo "Eval file already exists, skip BLEURT"
	# else
	srun python -m bleurt.score_files -candidate_file=${out} -reference_file=${ref} -bleurt_checkpoint $HOME/bleurt/BLEURT-20 -scores_file=${out}.eval.bleurt
	# fi

	# COMET (might not work so well for Maltese, as it is not in XLM-R)
	# if [[ -f "${out}.eval.comet" ]]; then
	# 	echo "Eval file already exists, skip COMET"
	# else
	comet-score -s $src -t $out -r $ref > ${out}.eval.comet
	# fi

	## BERT-score
	# First select the model based on the language
	# Highest scoring multi-lingual model (Maltese not in there)
	if [[ $lang = "mt" ]]; then
		# This model is 15G, can take quite a while to download
		model="google/mt5-xl" 
	else
		model="xlm-roberta-large" 
	fi

	# Now run the scoring
	# if [[ -f "${out}.eval.bertscore" ]]; then
	# 	echo "Eval file already exists, skip bert-score"
	# else
	bert-score --lang $lang -m $model -r $ref -c $out > ${out}.eval.bertscore
	# fi
fi

python /home1/s3412768/Genre-enabled-NMT/src/summarize.py \
    --folder $root_dir/eval/$exp_type/$model_type/$genre/ \
    --fname $out_file \
	--ref_with_tags $root_dir/data/MaCoCu.en-hr.test.tag.tsv \