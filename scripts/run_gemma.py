#!/usr/bin/env python3

import hashlib
import json
import os
import argparse
import pydantic
import sys
import time
from typing import List, Dict
from colorama import Fore, Back, Style
from google.api_core.exceptions import DeadlineExceeded

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)

for key in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
    if not os.environ.get(key):
        sys.exit(f"Missing key {key}")


def write_json(obj, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(obj, indent=4))


class JobsModel(pydantic.BaseModel):
    """
    We can use the json structure of this to inform gemma.
    In practice we didn't need Pydantic - we could have just
    written a json string I think.
    """

    application: str
    software: List[str]
    modules: List[str]
    environment_variables: Dict[str, str]
    resources: Dict[str, str]
    versions: Dict[str, str]


def model_to_json(model_instance):
    """
    Converts a Pydantic model instance to a JSON string.
    Args:
        model_instance (YourModel): An instance of your Pydantic model.
    Returns:
        str: A JSON string representation of the model.
    """
    return model_instance.model_dump_json()


# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
# !pip install -q -U keras-nlp
# !pip install -q -U keras>=3
# !pip install jax
# pip install -q -U google-generativeai colorama


# This can also be "tensorflow" or "torch".
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import keras
import keras_nlp
import fnmatch

# For inference, half-precision will work and save memory while mixed-precision is not applicable.
keras.config.set_floatx("bfloat16")


def content_hash(filename):
    with open(filename, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_file(filename):
    with open(filename, "r") as fd:
        content = fd.read()
    return content


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("code_gemma_2b_en")
    # gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.summary()

    # Example format for gemma
    json_model = model_to_json(
        JobsModel(
            application="lammps",
            software=[],
            modules=[],
            environment_variables={},
            resources={
                "gres": "",
                "cpus_per_task": "",
                "tasks": "",
                "ntasks_per_code": "",
                "gpus": "",
                "gpus_per_node": "",
                "cores_per_socket": "",
                "gpus_per_task": "",
                "exclusive": "",
                "cpus_per_gpu": "",
                "gpu_type": "",
                "time": "",
                "ntasks_per_node": "",
                "nodes": "",
                "memory": "",
                "sockets_per_node": "",
                "ntasks_per_socket": "",
                "mem_per_gpu": "",
                "mem_per_cpu": "",
                "gres_flags": "",
            },
            versions={},
        )
    )

    # Read in the jobspecs - not going to try for uniqueness this time
    contenders = list(recursive_find(args.input))

    seen = set()
    files = []
    for filename in contenders:
        digest = content_hash(filename)
        if digest in seen:
            continue
        seen.add(digest)
        files.append(filename)

    prompt = "Can you tell me what application this script is running and put the answer as a single term on the first line, followed by more detail about the other software and resource requirements in the script? And can you give me an output format in raw json\n"
    print("\n⭐️ The prompt is:")

    print(Fore.LIGHTRED_EX + prompt + Style.RESET_ALL)
    time.sleep(8)

    # Just for testing
    # IMPORTANT: uncomment and test a few before you run this at scale.
    # import IPython

    # IPython.embed()
    # sys.exit()

    total = len(files)
    print(f"Found {total} unique jobspec files.")
    for i, filename in enumerate(files):
        # If you want to take a sample
        # if i > 1000:
        #    break

        # Output files we want
        outfile = args.output + os.sep + os.path.relpath(filename, args.input)
        outfile_full = f"{outfile}-response.json"
        outfile_text = f"{outfile}-answer.json"
        if os.path.exists(outfile_full):
            continue

        print(
            Fore.LIGHTWHITE_EX
            + Back.GREEN
            + f"\nProcessing {filename}: {i} of {total}"
            + Style.RESET_ALL
        )
        content = read_file(filename)
        toprint = content

        optimized_prompt = (
            prompt
            + content
            + f". Please provide a response in a structured JSON format that matches the following model: {json_model}"
        )
        if len(content) > 300:
            toprint = content[0:150] + "\n...\n" + content[-150:]
        print(Fore.LIGHTBLUE_EX + toprint + Style.RESET_ALL)

        try:
            response = gemma_lm.generate(optimized_prompt)
            # response = model.generate_content(prompt + content)
        except DeadlineExceeded:
            print("Deadline exceeded - waiting to try another.")
            time.sleep(20)
            continue

        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if not response.parts:
            print(f"Warning: no response for {filename}")
            print(response)
            continue

        # Save both the full response, and raw json
        try:
            raw_json = response.parts[0].text.rsplit("```", 1)[0]
            raw_json = raw_json.replace("```json", "")
            raw_json = json.loads(raw_json)
            write_json(raw_json, outfile_text)
            print(Fore.LIGHTGREEN_EX + json.dumps(raw_json, indent=4) + Style.RESET_ALL)
        except:
            print(Fore.LIGHTGREEN_EX + response.parts[0].text + Style.RESET_ALL)
            pass

        write_json(response.to_dict(), outfile_full)
        print(Fore.LIGHTGREEN_EX + json.dumps(raw_json, indent=4) + Style.RESET_ALL)

    # This was start of how to do fine tuning.
    # gemma_lm.compile(sampler="top_k")
    # gemma_lm.generate("What is the meaning of life?", max_length=64)

    # TODO: to fine tune we will want to have an instruction and response
    # https://ai.google.dev/gemma/docs/lora_tuning#note_on_mixed_precision_fine-tuning_on_nvidia_gpus
    #
    # data = []
    # for filename in contenders:


#        features = json.loads(line)
#        # Filter out examples with context, to keep it simple.
#        if features["context"]:
#            continue
#        # Format the entire example as a single string.
#        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
#        data.append(template.format(**features))

# Only use 1000 training examples, to keep it fast.
# data = data[:1000]


def get_parser():
    parser = argparse.ArgumentParser(description="word2vec")
    parser.add_argument(
        "--input",
        help="Input directory",
        default=os.path.join(root, "data"),
    )
    parser.add_argument(
        "--output",
        help="Output data directory",
        default=os.path.join(here, "data", "gemma-with-template"),
    )
    return parser


if __name__ == "__main__":
    main()
