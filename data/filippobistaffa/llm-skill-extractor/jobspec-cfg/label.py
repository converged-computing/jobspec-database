import argparse as ap
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    # parse command-line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('--description', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'description.txt'))
    parser.add_argument('--model', type=str, default=os.path.join('models', 'mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf'))
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cmd', type=str, default='cmd.sh')
    args, additional = parser.parse_known_args()

    # build prompt
    with open(args.description) as f:
        description = f'' + ''.join(f.readlines()).strip()
    prompt = f'Which of the specialist tasks in the Australian Skills Framework are most related to the following course?'

    # format prompt according to the model
    model_formats = {
        'mixtral': '[INST] {prompt}\n{description} [/INST]',
        'vicuna': 'USER: {prompt}\n{description}\nASSISTANT:',
        'gemma': '<start_of_turn>user\n{prompt}\n{description}}<end_of_turn>\n<start_of_turn>model',
        'command-r': '## Instructions\n{prompt}\n## Input Text\n{description}',
    }
    for model, model_format in model_formats.items():
        if model in args.model:
            prompt_format = ('"' + model_format + '"').format(
                prompt = prompt,
                description = description
            ).replace("\n", "\\n").replace("\t", "\\t")

    # llama.cpp parameters
    llama_cpp_params = {
        '--model': args.model,
        '--ctx-size': str(args.ctx),
        '--seed': str(args.seed),
        '--prompt': prompt_format,
        '--repeat_penalty': '1.1',
        '--n-predict': '-1',
        '--temp': '0.7',
    }

    # build command-line
    llama_cpp_subdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'llama.cpp')
    command_line = [os.path.join(llama_cpp_subdir, 'build', 'bin' ,'main'), '--escape']
    for (param, value) in llama_cpp_params.items():
        command_line.extend([param, value])
    command_line.extend(additional) # by putting additional at the end we can override the default ones

    with open(args.cmd, "w") as f:
        f.write(" ".join(command_line))
