#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --job-name=pretrain-pixel
#SBATCH --ntasks=1 --cpus-per-task=48 --mem=70000M
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=75:00:00

nvidia-smi

export ENCODER="Team-PIXEL/pixel-base"
export DECODER="gpt2-large"
export DATASET="zuzannad1/pixelsum_wiki"
export EXPERIMENT_DIR="experiments/pretraining/$DECODER/"`date +%Y-%m-%d_%H-%M-%S`

mkdir -p ${EXPERIMENT_DIR}

python3 -m scripts.training.run_pretraining \
                   --encoder_name ${ENCODER} \
                   --decoder_name ${DECODER} \
                   --processor_name ${ENCODER} \
                   --tokenizer_name ${DECODER} \
                   --fallback_fonts_dir "fonts" \
                   --dataset_name ${DATASET} \
                   --remove_unused_columns false \
                   --dataloader_num_workers 32 \
                   --do_train true \
                   --do_eval true \
                   --do_predict true \
                   --train_decoder true \
                   --train_encoder true \
                   --evaluation_strategy "steps" \
                   --eval_steps 10000 \
                   --predict_with_generate true \
                   --logging_strategy "steps" \
                   --logging_steps 50 \
                   --save_strategy "steps" \
                   --save_steps 10000 \
                   --fp16 true \
                   --fp16_full_eval true \
                   --output_dir ${EXPERIMENT_DIR} \
                   --overwrite_output_dir false \
                   --log_predictions true \
                   --per_device_train_batch_size 16 \
                   --gradient_accumulation_steps 2 \
                   --learning_rate 1.5e-4 \
                   --weight_decay 0.05 \
                   --warmup_ratio 0.05 \
                   --lr_scheduler_type "cosine" \
                   --warmup_steps 15000 \
                   --max_steps 300000 \
                   --val_max_target_length 50 \
                   --max_target_length 50 \
                   --load_best_model_at_end true \
                   --use_fast_tokenizer true \
                   --num_beams 1 \
                   --report_to "wandb" \
                   --max_eval_samples 100 \
                   --data_cache_dir 'cached_data' \
                   