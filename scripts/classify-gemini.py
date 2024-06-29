#!/usr/bin/env/python

import sys
import os
import time
import json
import hashlib
import fnmatch
import argparse
from colorama import Fore, Back, Style
from typing import List, Dict
import pydantic


from google.api_core.exceptions import DeadlineExceeded

import google.generativeai as genai

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)

# pip install -q -U google-generativeai colorama


def model_to_json(model_instance):
    """
    Converts a Pydantic model instance to a JSON string.
    Args:
        model_instance (YourModel): An instance of your Pydantic model.
    Returns:
        str: A JSON string representation of the model.
    """
    return model_instance.model_dump_json()


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


token = os.environ.get("GEMINI_KEY")
if not token:
    sys.exit("export GEMINI_KEY to the environment.")
genai.configure(api_key=token)


def read_file(filename):
    with open(filename, "r") as fd:
        content = fd.read()
    return content


def write_json(obj, filename):
    import json

    with open(filename, "w") as fd:
        fd.write(json.dumps(obj, indent=4))


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
        default=os.path.join(here, "data", "gemini-with-template"),
    )
    return parser


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def content_hash(filename):
    with open(filename, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    # List models
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)

    # Testing case
    model = genai.GenerativeModel("gemini-1.5-flash")

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
                "ntasks_per_core": "",
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

    # We should try this:
    # https://hasanaboulhasan.medium.com/how-to-get-consistent-json-from-google-gemini-with-practical-example-48612ed1ab40
    # response = model.generate_content("What is the etymology of the phrase 'fast lane'?")

    parser = get_parser()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input):
        sys.exit("An input directory is required.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read in the jobspecs
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

    total = len(files)
    print(f"Found {total} unique jobspec files.")
    for i, filename in enumerate(files):
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
        if len(content) > 300:
            toprint = content[0:150] + "\n...\n" + content[-150:]
        print(Fore.LIGHTBLUE_EX + toprint + Style.RESET_ALL)

        optimized_prompt = (
            prompt
            + content
            + f". Please provide a response in a structured JSON format that matches the following model: {json_model}"
        )

        try:
            response = model.generate_content(optimized_prompt)
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

    import IPython

    IPython.embed()
    sys.exit()


if __name__ == "__main__":
    main()
