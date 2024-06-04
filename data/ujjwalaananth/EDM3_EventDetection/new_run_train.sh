#!/bin/bash
#SBATCH -n 20                        # number of cores
#SBATCH -t 0-20:00                  # wall time (D-HH:MM)
#SBATCH -N 1
##SBATCH -A uananthe
#SBATCH -p cidsegpu1
#SBATCH -q cidsegpu1_contrib_res
##SBATCH -q wildfire
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=uananthe@asu.edu # send-to address
#SBATCH -o /scratch/uananthe/ADD_t5_casie_text_ev_5121064.%j.out             # STDOUT (%j = JobId)
#SBATCH -e /scratch/uananthe/ADD_t5_casie_text_ev_5121064.%j.err             # STDERR (%j = JobId)

#export TRANSFORMERS_CACHE=/scratch/mparmar3/.cache/                                                           
#export PYTHONPATH=$PYTHONPATH:/home/mparmar3/.conda/envs/mihir_pytorch/lib/python3.7/site-packages/
#data_path="/scratch/mparmar3/artefacts/data"
#data_type='20_data'

module purge;
module load anaconda3/5.3.0;
source activate secenv;

#declare -a dataset_list=("Quoref")

python model2/run_model.py \
        --model_name_or_path 't5-base' \
        --do_train true \
        --train_file "data/CASIE_processed/train.csv" \
        --test_file "data/CASIE_processed/test.csv" \
        --output_dir "/scratch/uananthe/thesis/text_ev_casie_512_10_64" \
        --overwrite_output_dir true \
        --num_train_epochs="10" \
        --max_source_length="512" \
        --text_column="text" \
        --summary_column="event_triggers" \
        --per_device_train_batch_size="16" \
        --per_device_eval_batch_size="16" \
        --gradient_accumulation_steps="4" \
        --predict_with_generate \
        --save_strategy="steps" \
        --save_steps="500"

