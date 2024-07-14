import argparse
import json
import os
import random
import time
from typing import Any, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--truncate", type=int, default=16000)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--stop_idx", type=int, default=None)
    parser.add_argument("--verbose", default=True, action="store_false")
    return parser.parse_args(args)


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(model, tok: AutoTokenizer, input_text: str, max_tokens: int, args) -> str:
    """
    Truncate down to 128k then make inference.
    """
    verbose = args.verbose
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, args.truncate)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    inputs = tok(input_text, return_tensors="pt", truncation=False)
    inputs = inputs.to(model.device)  # type: ignore
    context_length = inputs.input_ids.shape[-1]
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=tok.pad_token_id,
        use_cache=True,
        min_length=context_length+1,
    )[0]
    output = tok.decode(output[context_length:], skip_special_tokens=True)
    print("Chunked generation:", output)
    return output


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    print("Time taken:", round(time.time() - start_time))
    return model, tok  # type: ignore


if __name__ == "__main__":
    args = parse_args()

    print(json.dumps(vars(args), indent=4))
    datasets = [
        "code_debug",
        "code_run",
        "kv_retrieval",
        "longbook_choice_eng",
        "longbook_qa_chn",
        "longbook_qa_eng",
        "longbook_sum_eng",
        "longdialogue_qa_eng",
        "math_calc",
        "math_find",
        "number_string",
        "passkey",
    ]

    # Model
    model, tok = load_model(args.model)
    for data_name in datasets:
        max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]

        # Data
        result_dir = f"pred/{args.model}_{args.truncate}_v2"
        os.makedirs(result_dir, exist_ok=True)
        examples = load_data(data_name, data_dir="data")

        if args.stop_idx is None:
            args.stop_idx = len(examples)
            output_path = f"{result_dir}/{data_name}.jsonl"
        else:
            output_path = f"{result_dir}/preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa

        if os.path.exists(output_path):
            os.remove(output_path)

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Start index: {args.start_idx}")
        print(f"Stop index: {args.stop_idx}")
        print(f"Verbose: {args.verbose}")
        print(f"Max tokens: {max_tokens}")
        for i in range(args.start_idx, args.stop_idx):
            eg = examples[i]
            input_text = create_prompt(eg, data_name, args.model, "data")
            print(f"====== Example {i} ======")
            pred = get_pred(model, tok, input_text, max_tokens=max_tokens, args=args)
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)
        args.stop_idx = None
