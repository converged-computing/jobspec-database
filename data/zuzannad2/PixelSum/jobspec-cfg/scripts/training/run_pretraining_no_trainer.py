#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.
import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb 
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    HfArgumentParser,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from pixel import (
    PangoCairoTextRenderer,
    PyGameTextRenderer,
    get_transforms,
    get_attention_mask
)

from PIL import Image

from transformers.utils.versions import require_version
from pixelsum.modeling_pixelsum import PIXELSumModel
from schemas.custom_args import DataTrainingArguments, TrainingArguments, ModelArguments

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.34.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

wandb.init(project="pixelsum")

def get_renderer(model_args: argparse.Namespace):
    renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
    renderer = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
        )
    
    return renderer

def get_tokenizer(model_args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.decoder_name,
            use_fast=model_args.use_fast_tokenizer,
            add_prefix_space=True if model_args.decoder_name == "gpt2" else False,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    tokenizer.pad_token = '<|pad|>'

    return tokenizer


def get_model_and_config(model_args: argparse.Namespace):  

    model = PIXELSumModel.from_encoder_decoder_pretrained(
            model_args.encoder_name,
            model_args.decoder_name,
            cross_attention_reduce_factor=1
        )
    if model_args.train_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = True


    if "opt" in model_args.decoder_name:
        if not model_args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'encoder_attn' not in name:
                    param.requires_grad = True

    elif "gpt" in model_args.decoder_name:
        if not model_args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad = False
   
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    
    logger.info('Training a model with {} trainable parameters.'.format(num_trainable_params))
    logger.info(f"Using dropout with probability {model_args.dropout_prob}")
    
    return model, model.config

def main():
    # Parse the arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))  
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=training_args.report_to, project_dir=training_args.output_dir, mixed_precision='fp16') if training_args.with_tracking else Accelerator()
    )

    # if accelerator.is_main_process:
    #     wandb.init(project="pixelsum")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    dataset = load_dataset("zuzannad1/pixelsum_wiki", split="train", cache_dir=data_args.data_cache_dir)
    
    # Initialise model and processors
    model, config = get_model_and_config(model_args)
    renderer = get_renderer(model_args)
    tokenizer = get_tokenizer(model_args)
    
    model.config.vocab_size = model.config.decoder.vocab_size
    # Set decoder_start_token_id
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    model.config.is_encoder_decoder = True

    accelerator.init_trackers(
        project_name="pixelsum", 
        config=config
    )

    transforms = get_transforms(
        do_resize=True, 
        size=(renderer.pixels_per_patch, renderer.pixels_per_patch * renderer.max_seq_length))
    
    def preprocess_examples(batch):
        docs, summaries = batch['example'], batch['summary']
        data = {"pixel_values": [], "attention_mask": [], "label_ids": []}

        def _pad_input_ids(input_ids, max_length=data_args.max_target_length):
            input_ids += [-100] * (max_length - len(input_ids))
            return input_ids
        
        for document, summary in zip(docs, summaries):
            encoding = renderer(document)
            image = encoding.pixel_values
            num_patches = encoding.num_text_patches
            
            pixel_values = transforms(Image.fromarray(image))
            attention_mask = get_attention_mask(num_patches, seq_length=529)

            text_ids = tokenizer.encode(summary)
            text_ids = text_ids[:data_args.max_target_length] # Truncate
            input_ids = _pad_input_ids(text_ids) # Pad - disabling pad to see this other collator
        
            assert len(attention_mask) == 529
            
            data["pixel_values"].append(torch.tensor(pixel_values))
            data["attention_mask"].append(torch.tensor(attention_mask))
            data["label_ids"].append(torch.tensor(input_ids))

        return data
        

    split_dataset = dataset.train_test_split(test_size=0.1)
    
    train_dataset = split_dataset["train"]
    train_dataset.set_transform(preprocess_examples)
    eval_dataset = split_dataset["test"]
    eval_dataset.set_transform(preprocess_examples)
    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = training_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            checkpoint_path = training_args.resume_from_checkpoint
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        if training_args.with_tracking:
            total_loss = 0
        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if training_args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if isinstance(training_args.logging_steps, int):
                if completed_steps % training_args.logging_steps == 0:
                    local_step = int(completed_steps / training_args.logging_steps)
                    accelerator.log({"train/loss": loss}, step=local_step)
                    accelerator.log({"train/learning_rate": lr_scheduler.get_last_lr()[0]}, step=local_step)
                    #accelerator.log({"train/epoch": completed_steps / training_args.max_train_steps}, step=local_step)
                    accelerator.log({"train/global_step": completed_steps}, step=local_step)

            if completed_steps >= training_args.max_train_steps:
                break

        model.eval()

        if training_args.push_to_hub and epoch < training_args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(training_args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if training_args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if training_args.output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if training_args.with_tracking:
        accelerator.end_training()

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
        

if __name__ == "__main__":
    main()