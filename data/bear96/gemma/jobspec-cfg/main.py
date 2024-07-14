import argparse
import numpy as np
import torch
import sys 
from config import GemmaConfig, get_config_for_7b, get_config_for_2b
from tokenizer import Tokenizer
import contextlib
import os
from model import GemmaForCausalLM
from heatmaps import generate
import json


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)
    
def main():
    VARIANT = "7b-it"
    weights_dir = "gemma-ckpt"
    model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    if not os.path.exists(weights_dir):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it-pytorch", token = "")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it-pytorch", token = "")
        ckpt_path = model.state_dict()
    else:    
        model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
        model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device: ", device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = GemmaForCausalLM(model_config)
        ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
        model.load_weights(ckpt_path)
        model = model.to(device).eval()
    
    USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
    MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

    covid = """
    You are Brandy. You are 28 years old. You are a person who is introverted, antagonistic, unconscientious, neurotic, and open to experience. \
    You live in the town of Dewberry Hollow. You have a job and go to the office for work everyday. \
    You have a cough and a fever. You go to work to earn money to support yourself. \
    You know about the Catasat virus spreading across the country. It is an airborne virus causing an infectious disease that spreads from human to human. The deadliness of the virus is unknown. \
    You check the newspaper and find that 5% of Dewberry Hollow's population were diagnosed with new infections of the Catasat virus yesterday. \
    Do you want to stay at home for the entire day? Respond with 'Yes' if you want to stay at home or 'No' if you want to go out.
    """

    prompt = (
        USER_CHAT_TEMPLATE.format(
            prompt=covid
        )
        + "<start_of_turn>model\n"
    )

    result, tokens, attention_weights = model.generate(
        prompt,
        device=device,
        output_len=100,
    )
    print(result)
    input_len = len(model.tokenizer.encode(prompt))
    total_len = len(model.tokenizer.encode(prompt)) + len(model.tokenizer.encode(result))
    total_tokens = model.tokenizer.encode(prompt) + model.tokenizer.encode(result)
    total_tokens = [model.tokenizer.decode(v) for v in total_tokens]
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    with open("outputs/output_tokens.txt", 'w') as f:
    	for tok in total_tokens:
    		print(tok, file=f)
            
    
    with open("outputs/attention_weights.json","w") as f:
        json.dump(attention_weights, f, sort_keys=True, indent=4)
    for key, val in attention_weights.items():
        # Convert each element of val to torch.tensor with dtype=torch.float16
        val = [torch.tensor(v, dtype=torch.float16) for v in val]
        # Concatenate tensors along dimension 2
        val = torch.cat(val, dim=2)
        attention_weights[key] = val[:, :, :len(total_tokens), :len(total_tokens)]

    for key, val in attention_weights.items():
        attention_weights[key] = val[:, :, :len(total_tokens), :len(total_tokens)]

    include_layers = [-1]
    attention = format_attention(list(attention_weights.values()), include_layers)
    tokens = total_tokens
                     
    sentence_b_start = input_len+1
    slice_a = slice(0, sentence_b_start)
    slice_b = slice(sentence_b_start, len(tokens))
    attn = attention[:, :, slice_b, slice_a]
    left = tokens[slice_b]
    right = tokens[slice_a]
    for i in range(8):
        No_attn= attn[:,i,0,:] # (batch, heads, in_seq, out_seq)
        flat_attn = No_attn.flatten()
        generate(
            text_list = right,
            attention_list = flat_attn.tolist(), 
            latex_file = f"outputs/GemmaDecoderLayer-18-Head-{i+1}.tex", 
            rescale_value = True
        )
        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt", type=str, required=True)
    # parser.add_argument("--variant",
    #                     type=str,
    #                     default="2b",
    #                     choices=["2b", "7b"])
    # parser.add_argument("--device",
    #                     type=str,
    #                     default="cpu",
    #                     choices=["cpu", "cuda"])
    # parser.add_argument("--output_len", type=int, default=100)
    # parser.add_argument("--seed", type=int, default=12345)
    # parser.add_argument("--quant", action='store_true')
    # parser.add_argument("--prompt", type=str, default="The meaning of life is")
    # args = parser.parse_args()

    main()
