#!/usr/bin/env/python

# A script to visually compare between cases.
# pip install rich

import argparse
from rich.console import Console
from rich.markdown import Markdown
import os
import json
import sys
import random

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)


def read_file(filename):
    with open(filename, "r") as fd:
        content = fd.read()
    return content


def get_parser():
    parser = argparse.ArgumentParser(description="visualize Gemini")
    parser.add_argument(
        "--type",
        help="Visualize a subset of 'wrong' or 'missing' (defaults to 'wrong')",
        default="wrong",
        choices=["wrong", "missing"],
    )
    parser.add_argument(
        "-n",
        "--number",
        help="Number to view (defaults to 1)",
        default=1,
        type=int,
    )
    return parser


# Lookup of files
data_dir = os.path.join(here, "data", "gemini-with-template-processed")
lookup = {
    # This was too big for GitHub
    #    "all": os.path.join(data_dir, "visual-resource-comparison.json"),
    "missing": os.path.join(data_dir, "visual-with-missing-resource-comparison.json"),
    "wrong": os.path.join(data_dir, "visual-with-wrong-resource-comparison.json"),
}


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if args.number < 1:
        sys.exit("Please choose a number >= 1 you squirrel.")

    filename = lookup[args.type]
    records = json.loads(read_file(filename))
    console = Console()

    for i in range(args.number):
        idx = random.choice(list(records.keys()))
        record = records[idx]

        # This was a typo I mde
        record = record.replace("#", "# ", 1)
        console.print(Markdown(record))


if __name__ == "__main__":
    main()
