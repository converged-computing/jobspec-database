
import wandb
import os
import yaml
from pprint import pprint
import argparse

from config.config import TrainingConfig
import pandas as pd
import numpy as np
import torch

from utils.dataset import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer

def merge_quantized_model(train_args):

    trained_model = AutoPeftModelForCausalLM.from_pretrained(
        train_args.output_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )

    return trained_model.merge_and_unload()


def prepare_trainer(model,data,tokenizer,train_args):
    
    lora_r = train_args.lora.lora_r
    lora_alpha = train_args.lora.lora_alpha
    lora_dropout = train_args.lora.lora_dropout
    lora_target_modules = train_args.lora.lora_target_modules


    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=train_args.task_type,
    )

    training_arguments = TrainingArguments(
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        gradient_checkpointing = train_args.gradient_checkpointing,
        eval_accumulation_steps=train_args.eval_accumulation_steps,
        optim=train_args.optim,
        logging_steps=train_args.logging_steps,
        learning_rate=train_args.learning_rate,
        fp16=train_args.fp16,
        max_grad_norm=train_args.max_grad_norm,
        num_train_epochs=train_args.num_train_epochs,
        evaluation_strategy=train_args.evaluation_strategy,
        eval_steps=train_args.eval_steps,
        warmup_ratio=train_args.warmup_ratio,
        save_strategy=train_args.save_strategy,
        group_by_length=True,
        output_dir=train_args.output_dir,
        report_to=train_args.report_to,
        save_safetensors=True,
        lr_scheduler_type=train_args.lr_scheduler,
        seed=train_args.seed,
    )

    # %%
    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    return trainer

def create_model_and_tokenizer(model_args,tokenizer_args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.name,
        use_safetensors=True,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_args.model_name)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'context' %}\n{{ '<|context|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def prepare_logging(experiment):

    wandb.login(key=experiment.wandb_token)
    os.environ["WANDB_PROJECT"]= experiment.name
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="false"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"
    print("Wandb Loggin Successfull")


def main(config_path:str):
    with open(config_path,'r') as f:
        config = TrainingConfig(**yaml.safe_load(f))
    
    prepare_logging(config.experiment)
    
    data = Dataset(config.dataset,config.tokenizer,config.train.seed).prepare_dataset()
    model, tokenizer = create_model_and_tokenizer(config.model,config.tokenizer)
    model.config.use_cache = False
    print(model.config.quantization_config.to_dict())

    trainer = prepare_trainer(model,data,tokenizer,config.train)
    trainer.train()
    trainer.save_model()

    model = merge_quantized_model(config.train)
    model.save_pretrained(config.experiment.model_save_name, safe_serialization=True)
    tokenizer.save_pretrained(config.experiment.model_save_name)

    if config.experiment.push_to_hub:
        model.push_to_hub(config.experiment.model_save_name)
        tokenizer.push_to_hub(config.experiment.model_save_name)

if __name__=="__main__":
   
    parser = argparse.ArgumentParser(description="Supervised Fine Tuning")
    parser.add_argument("--config",dest="config", required=True,type=str,action='store')

    args = parser.parse_args()
    main(args.config)

   
    

