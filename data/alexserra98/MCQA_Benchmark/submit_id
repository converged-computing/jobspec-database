#!/bin/bash
#SBATCH --partition=DGX
##SBATCH --nodelist=dgx002
#SBATCH --account=lade
#SBATCH --nodes=1
#SBATCH --time=6:00:00            
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32           
#SBATCH --mem=100G                
#SBATCH --job-name=test
#SBATCH --gres=gpu:1


#source /etc/profile.d/modules.sh
#module use /opt/nvidia/hpc_sdk/modulefiles/
#module load nvhpc

source /u/area/ddoimo/anaconda3/bin/activate ./env_amd

#source /u/area/ddoimo/anaconda3/bin/activate /u/area/ddoimo/ddoimo/open/open-instruct/env_amd

export OMP_NUM_THREADS=32

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

#export CUDA_LAUNCH_BLOCKING=1

#model_name=llama-2-7b
#path=llama_v2

model_name=mistral-1-7b
path=mistral_v1

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
       	--nnodes=1  --nproc-per-node=1 \
    diego/extraction/extract_repr.py \
    --checkpoint_dir  "/u/area/ddoimo/ddoimo/models/$path/models_hf/$model_name" \
    --use_slow_tokenizer \
    --preprocessing_num_workers 16 \
    --micro_batch_size 1 \
    --out_dir "./results" \
    --logging_steps 100 \
    --layer_interval 1 \
    --save_distances --save_repr \
    --remove_duplicates \
    --use_last_token \
    --max_seq_len 4090 \
    --split "dev+validation" \
    --num_few_shots 0 \
    --finetuned_path  "/u/area/ddoimo/ddoimo/finetuning_llm/open-instruct/results" \
    --finetuned_mode "dev_val_balanced_20samples" \
    --finetuned_epochs 4 \
    --step $step
    #--ckpt_epoch 4  
    #--split "test" \
    #--prompt_search
    #--random_order \
    #--skip_choices 
    #--wrong_answers  
    #--declarative 
    #--random_subject
    #--declarative
    #--sample_questions
    #--declarative \
    #--aux_few_shot 
    #--prompt_search \
    #--aux_few_shot  
    #--sample_questions
    # --finetuned_path  "/u/area/ddoimo/ddoimo/finetuning_llm/open-instruct/results" \
    # --finetuned_mode "test_balanced" \
    # --finetuned_epochs 4 \
    # --ckpt_epoch 3 
