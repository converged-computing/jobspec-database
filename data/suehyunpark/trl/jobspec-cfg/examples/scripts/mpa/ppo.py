# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
python examples/scripts/ppo.py \
    --log_with=wandb
"""
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from tqdm import tqdm
import bitsandbytes as bnb
from transformers import HfArgumentParser, AutoModelForSequenceClassification

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, ModelConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from get_module import get_model, get_tokenizer

tqdm.pandas()


@dataclass
class DataConfig:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    max_length: int = field(
        metadata={"help": "Maximum length of the input sequence."}
    )
    repo_id: str = field(
        metadata={"help": "ID of the repository."}
    )
    run_name: str = field(
        metadata={"help": "Run name of wandb"}
    )
    save_steps: int = field(
        metadata={"help": "Number of steps to save the model."}
    )

def preprocess_dataset(ppo_config, tokenizer, data_config):
    raw_dataset = load_dataset('json', data_files=ppo_config.query_dataset, split="train")
    original_columns = raw_dataset.column_names
    
    def apply_template(system, instruction):
        # Mistral template
        prompt = f"{system}\n{instruction}".strip()
        return f"[INST] {prompt} [/INST] "
    
    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": []
        }
        for system, instruction in zip(examples["system"], examples["instruction"]):
            query = apply_template(system, instruction)
            tokenized_query = tokenizer(query, max_length=data_config.max_length, truncation=True)

            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_query["input_ids"])

        return new_examples

    preprocessed_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        remove_columns=original_columns
    )
    
    # https://github.com/huggingface/trl/issues/1473#issuecomment-2078495703
    # When the prompt exceeds the max_length, the log probabilities for both chosen and rejected turn to NaN. 
    # Consider filtering out cases where the prompt is longer than the max_length or max_prompt_len. 
    # The reason for trimming cases where the prompt exceeds max_prompt_len is that 
    # if the chosen or rejected segments are significantly shorter than the prompt, it may hinder effective learning.
    preprocessed_dataset = preprocessed_dataset.filter(
        lambda x: len(x["input_ids"]) <= data_config.max_length
    )
    
    return preprocessed_dataset

# mini batch size 1 for each generation
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

def main(ppo_config, model_config, data_config):
    ppo_config.tracker_kwargs = {"wandb": {"name": data_config.run_name}}
    ppo_config.push_to_hub_if_best_kwargs = {"repo_id": data_config.repo_id, "private": True}
    set_seed(ppo_config.seed)
    
    model = get_model(model_config, AutoModelForCausalLMWithValueHead)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    
    tokenizer = get_tokenizer(model_config)
    # https://github.com/huggingface/trl/issues/937#issuecomment-1793697802
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    
    ref_model = get_model(model_config, AutoModelForCausalLMWithValueHead)
    
    train_dataset = preprocess_dataset(ppo_config, tokenizer, data_config)
    
    adam = bnb.optim.Adam8bit(model.parameters(), lr=ppo_config.learning_rate)

    ppo_trainer = PPOTrainer(
        ppo_config, 
        model, 
        ref_model, 
        tokenizer, 
        dataset=train_dataset,
        data_collator=collator,
        optimizer=adam
    )
    
    # `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
    model_config.model_name_or_path = ppo_config.reward_model
    reward_model = get_model(model_config, AutoModelForSequenceClassification, num_labels=1)
    
    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    output_length_sampler = LengthSampler(32, data_config.max_length)
    
    for steps, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
        # https://github.com/huggingface/trl/issues/787#issuecomment-1937917675
        query_tensors = [torch.LongTensor(query_tensor) for query_tensor in query_tensors]
        # Get response
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, 
            return_prompt=False, 
            generate_ref_response=True, 
            length_sampler=output_length_sampler,
            **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)
        
        # Compute reward for the responses
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        input_ids = tokenizer(texts, max_length=data_config.max_length, padding=True, truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(reward_model.device)
        reward_outputs = reward_model(input_ids)[0]
        rewards = [torch.tensor(output.item()) for output in reward_outputs]
        batch["rewards"] = rewards
        
        # Compute reward for the reference responses
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_input_ids = tokenizer(ref_texts, max_length=data_config.max_length, padding=True, truncation=True, return_tensors="pt").input_ids
        ref_input_ids = ref_input_ids.to(reward_model.device)
        ref_reward_outputs = reward_model(ref_input_ids)[0]
        ref_rewards = [torch.tensor(output.item()) for output in ref_reward_outputs]
        batch["ref_rewards"] = ref_rewards
        
        # Run PPO step
        stats = ppo_trainer.step(
            query_tensors, 
            response_tensors, 
            rewards
        )
        
        ppo_trainer.log_stats(
            stats,
            batch,
            rewards,
            columns_to_log=["query", "response", "ref_response", "rewards", "ref_rewards"]
        )

        if steps % ppo_config.save_steps == 0:
            ppo_trainer.save_model(data_config.output_dir)
            
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = HfArgumentParser((PPOConfig, ModelConfig, DataConfig))
    ppo_config, model_config, data_config = parser.parse_args_into_dataclasses()
    
    main(ppo_config, model_config, data_config)
    