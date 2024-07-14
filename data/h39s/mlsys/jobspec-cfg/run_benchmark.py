import os
import sys
import json
import time
import yaml
from datetime import datetime
import itertools

sys.path.append("./")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from transformers import AutoConfig, AutoTokenizer
import time
from tqdm import tqdm
from src.build_model import OffloadConfig, QuantConfig, build_model
from transformers import TextStreamer
import numpy as np

run_config = yaml.safe_load(open(sys.argv[1]))
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
state_path = "./Mixtral-8x7B-Instruct-v0.1-offloading-demo"
benchmark_prompts = "benchmark_prompts.txt"


def read_prompt(file_path):
    with open(file_path, "r") as f:
        prompts = f.readlines()
    return prompts


all_prompts = read_prompt(benchmark_prompts)
if run_config.get("num_prompts", False):
    all_prompts = all_prompts[: run_config["num_prompts"]]
print(run_config)


for offload_per_layer, cache_strategy, max_seq_len in itertools.product(
    run_config["offload_per_layer"], run_config["cache_strategy"], run_config["max_seq_len"]
):
    print(
        f"Running benchmark for cache_strategy: {cache_strategy} and max_seq_len: {max_seq_len}"
    )
    config = AutoConfig.from_pretrained(quantized_model_name)

    device = torch.device("cuda:0")
    # offload_per_layer = run_config["offload_per_layer"]

    num_experts = config.num_local_experts
    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    sequence = None
    track_runs_time = []
    track_runs_num_tokens = []
    track_runs_tokens_per_second = []
    track_runs_hits = []

    dir_prefix = f"logs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    log_dir = f"{dir_prefix}_{run_config['gpu']}_{run_config['offload_per_layer']}_{cache_strategy}_{max_seq_len}"
    os.makedirs(log_dir, exist_ok=True)
    # dump config to log_dir
    with open(f"{log_dir}/config.yaml", "w") as f:
        yaml.dump(run_config, f)

    for run_idx in range(run_config["num_runs"]):
        total_time = []
        total_num_tokens = []
        print(f"Running benchmark for run {run_idx}")
        if "offload_json" in run_config:
            print("Loading experts to offload from json")
            with open(run_config["offload_json"], "r") as f:
                experts_to_offload = json.load(f)
                # convert keys and values to int
                experts_to_offload = {
                    int(k): [int(exp) for exp in v]
                    for k, v in experts_to_offload.items()
                }
                model, expert_cache_obj = build_model(
                    device=device,
                    quant_config=quant_config,
                    offload_config=offload_config,
                    state_path=state_path,
                    cache_strategy=cache_strategy,
                    experts_to_offload=experts_to_offload,
                )
        else:
            model, expert_cache_obj = build_model(
                device=device,
                quant_config=quant_config,
                offload_config=offload_config,
                state_path=state_path,
                cache_strategy=cache_strategy,
            )
        run_log_dir = f"{log_dir}/run_{run_idx}"
        os.makedirs(run_log_dir, exist_ok=True)
        seq_len = 0
        # CHANGE FILENAME HERE

        for i in range(len(all_prompts)):
            start = time.time()
            print("User: ", end="")
            user_input = all_prompts[i]
            print(user_input)
            print("\n")

            user_entry = dict(role="user", content=user_input)
            input_ids = tokenizer.apply_chat_template(
                [user_entry], return_tensors="pt"
            ).to(device)

            attention_mask = torch.ones_like(input_ids)
            print("Mixtral: ", end="")
            result = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                max_new_tokens=max_seq_len,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
            print("\n")
            sequence = result["sequences"]
            end = time.time()
            total_time.append(end - start)
            seq_len = sum([len(seq) for seq in sequence])
            total_num_tokens.append(seq_len)

        filename = "results"
        with open(f"{run_log_dir}/{filename}.txt", "w") as log_file:
            print("TIME BENCHMARKS", file=log_file)
            print(f"Total time taken: {sum(total_time)} seconds", file=log_file)
            print(
                f"Total number of tokens generated: {sum(total_num_tokens)}",
                file=log_file,
            )
            print(
                f"Average token per second: {sum(total_num_tokens)/sum(total_time)}",
                file=log_file,
            )
            track_runs_time.append(sum(total_time))
            track_runs_num_tokens.append(sum(total_num_tokens))
            track_runs_tokens_per_second.append(sum(total_num_tokens) / sum(total_time))
            print("\n\n\n", file=log_file)

            print("HIT RATE BENCHMARKS", file=log_file)
            data_hits = {}

            for k in expert_cache_obj.group_infos:
                data_hits[k] = expert_cache_obj.group_infos[k].expert_counts
            # print(data_hits)
            # print overall hit rate and hit rate per layer
            overall_hits = 0
            overall_misses = 0
            for layer in data_hits:
                tot_calls = 0
                tot_hits = 0
                # print(data_hits[layer])
                for exp in data_hits[layer]:
                    tot_calls += data_hits[layer][exp][0]
                    tot_hits += data_hits[layer][exp][1]
                # print(tot_hits, tot_calls)
                overall_hits += tot_hits
                overall_misses += tot_calls - tot_hits
                print(f"Layer {layer}: Hit rate = {tot_hits/tot_calls}", file=log_file)

            print(
                f"Overall hit rate = {overall_hits/(overall_hits + overall_misses)}",
                file=log_file,
            )
            track_runs_hits.append(overall_hits / (overall_hits + overall_misses))

        with open(f"{run_log_dir}/{filename}.json", "w") as dump_data_file:
            # dump data_hits, total_time, total_num_tokens to a json file
            import json

            all_stats = {
                "data_hits": data_hits,
                "total_time": total_time,
                "total_num_tokens": total_num_tokens,
            }
            json.dump(all_stats, dump_data_file, indent=4)

        del model
        torch.cuda.empty_cache()
        time.sleep(5)

    with open(f"{log_dir}/overall_results.txt", "w") as overall_results_file:
        print("OVERALL RESULTS", file=overall_results_file)
        print(f"All times", track_runs_time, file=overall_results_file)
        print(f"All num tokens", track_runs_num_tokens, file=overall_results_file)
        print(
            f"All tokens per second",
            track_runs_tokens_per_second,
            file=overall_results_file,
        )
        print(f"All hits", track_runs_hits, file=overall_results_file)

        # print mean and std of all times, num tokens, tokens per second, hits as mean
        # +- std
        print("OVERALL STATS", file=overall_results_file)
        print(
            f"Run_Time: {sum(track_runs_time)/len(track_runs_time)} +- {np.std(track_runs_time)}",
            file=overall_results_file,
        )
        print(
            f"Num Tokens: {sum(track_runs_num_tokens)/len(track_runs_num_tokens)} +- {np.std(track_runs_num_tokens)}",
            file=overall_results_file,
        )
        print(
            f"Tokens per second: {sum(track_runs_tokens_per_second)/len(track_runs_tokens_per_second)} +- {np.std(track_runs_tokens_per_second)}",
            file=overall_results_file,
        )
        print(
            f"Hit Rate: {sum(track_runs_hits)/len(track_runs_hits)} +- {np.std(track_runs_hits)}",
            file=overall_results_file,
        )
