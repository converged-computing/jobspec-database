from lm_eval import evaluator
from lm_eval.models.huggingface import AutoCausalLM
from lm_eval.api.registry import get_model

import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoTokenizer
import torch
import time
import utils

device = "cuda:0"
task_list=["headqa_en","headqa_es"]
# task_list=["headqa_en","headqa_es","pawsx_en","pawsx_es","pawsx_zh","xnli_en","xnli_es","xnli_zh", "xstory_cloze_en", "xstory_cloze_es", "xstory_cloze_zh"]
batch_size = 1
max_batch_size = 64

if True:    


    model_path = "/home/data_shares/mapillary/llama/train/minillm_init/llama-7B"
    model_args = f"pretrained={model_path},cache_dir=./llm_weights"
    
    
    # create_from_arg_string(cls, arg_string, additional_config=None): Need to put pretrained model in additional_config i.e pretrained:model
    start_time = time.time()
    lm = get_model("hf").create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    # "device": device,
                },
            )
    print("time to load model:", time.time() - start_time)





# lm = get_model("hf-causal")(
#             pretrained=model,
#             tokenizer=tokenizer,
#             batch_size=batch_size,
#             max_batch_size=max_batch_size,
#         )


# lm = get_model("hf-causal").create_from_arg_string(
#         "",
#         {
#             "batch_size": 1,
#             "max_batch_size": 8,
#             "device": device,
#             "tokenizer": True,
#             "pretrained": model,
#             "tokenizer": AutoTokenizer.from_pretrained("gpt2-large"),
#         },
#     )

# Should probably be the way to do things
# TODO figure out if device is also needed maybe also tokenizer


# model = AutoModel.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model_name = "7B-init-13B-sft" 
# model = get_llm(model_name, cache_dir="llm_weights")
# print("model", model)
# start_time = time.time()
# checksum = create_checksum(model)
# print("time to create checksum:", time.time() - start_time)


def eval(model, task_list, device):
    start_time = time.time()
    results = evaluator.simple_evaluate(
            model=model,
            # model_args=model_args,
            tasks=task_list,
            device=device,
            # pretrained_model=model,
            # tokenizer=tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            
        )
    print("time to evaluate:", time.time() - start_time)
    return results



# model = AutoModelForCausalLM.from_pretrained('gpt2-large')
# utils.save_results("GPT2-774M_preloaded", eval(model, task_list, device))

# # del model
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# utils.save_results("BLOOM-560M", eval(model, task_list, device))


model_name = "bigscience/bloom-560m"

lm = get_model("hf-causal-experimental")(
            pretrained=model_name,
            # batch_size=batch_size,
            # max_batch_size=max_batch_size,
        )

results = evaluator.simple_evaluate(
            model=lm,
            model_args="",
            tasks=task_list,
            device=device,
            # pretrained_model=model,
            # tokenizer=tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            write_out=True,)
utils.save_results("GPT2-774M_all_tasks", results)