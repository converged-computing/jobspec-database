MODEL_SIZE=$1
# MODEL=/gpfs/u/home/AICD/AICDzhqn/scratch-shared/llama_zf/llama-2-${MODEL_SIZE}-hf/
MODEL=/vast/work/public/ml-datasets/llama-2/Llama-2-${MODEL_SIZE}b-chat-hf
# !torchrun --nproc_per_node 1 run_ds_lora.py \
# --bf16 True \
# --tf32 True \
deepspeed erasure/train_ds_lora.py \
  --model_id ${MODEL} \
  --dataset_path erasure/bio-processed-with-indices \
  --output_dir erasure/llama13b-lora-with-indices \
  --num_train_epochs 1 \
  --bf16 True \
  --tf32 True \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-3 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 2 \
  --use_flash_attn True \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 25 \
  --save_steps 100 \
  --save_total_limit 3 \
  --deepspeed configs/llama_ds.json
