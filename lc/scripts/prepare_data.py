#!/usr/bin/env python3

# Prepare data will create a view of the extracted data from raw
# and put just the organized job files, by cluster,
# in one location.

# chmod ugo+rwx
import tarfile
import argparse
import fnmatch
import hashlib
import os
import re
import sys
import json

import rse.utils.file as utils


here = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(here)


def get_parser():
    parser = argparse.ArgumentParser(description="word2vec")
    parser.add_argument(
        "--input",
        help="Input directory",
        default=os.path.join(root, "raw", "jobdata_json"),
    )
    parser.add_argument(
        "--output",
        help="Output data directory",
        default=os.path.join(root, "data", "jobspecs"),
    )
    return parser


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def combine_data(data_file, args, files):
    seen = set()
    with open(data_file, "w") as fd:
        for file in files:
            print(f"Adding jobspec {file}")
            # 1. Step 1, ensure we don't duplicate files
            digest = content_hash(file)
            if digest in seen:
                print(f"Found duplicate file {file}")
                continue
            seen.add(digest)
            content = utils.read_file(file)
            tokens = tokenize(content)

            # Write each "document" on a single line
            fd.write(" ".join(tokens) + "\n")


# This gets replaced with a space (to make new tokens)


def tokenize(lines):
    """
    Special tokenize for scripts, and parsing of flag directives
    """
    punctuation = "(/|'|\"|;|:|=|\n)"
    # This gets removed
    removed = "({|}|[|]|[)]|[(]|[]]|[[]|[$]|[#])"
    content = re.sub(punctuation, " ", lines)
    content = re.sub(removed, "", content)
    line = content.lower().split()
    line = [x for x in line if len(x) > 1]
    return line


def content_hash(content):
    h = hashlib.new("sha256")
    try:
        h.update(content.encode())
    except:
        h.update(content)
    return h.hexdigest()


def main():
    """
    Preprocessing for word2vec.
    """
    parser = get_parser()
    args, _ = parser.parse_known_args()
    print(args.input)
    if not os.path.exists(args.input):
        sys.exit(f"{args.input} does not exist.")

    # Keep a count!
    digests = set()
    duplicates = 0

    fd = open("jobspec-corpus.txt", "w")
    meta = open("jobspec-meta.txt", "w")

    # First process json files
    files = list(recursive_find(args.input, "*.json"))
    total = len(files)
    for i, filename in enumerate(files):
        print(f"Processing {i} of {total}", end="\r")
        jobid = os.path.basename(filename).replace(".json", "")
        content = utils.read_json(filename)
        digest = content_hash(json.dumps(content))
        if digest in digests:
            duplicates += 1
            continue

        if "BatchScript" not in content["scontrol"]:
            continue

        script = content["scontrol"]["BatchScript"]
        tokens = tokenize(script)
        fd.write(" ".join(tokens) + "\n")
        meta.write(f"{filename} {jobid}\n")

    # Now read the tars
    tarfiles = list(recursive_find(args.input, "*.tar"))
    total = len(tarfiles)
    for i, filename in enumerate(tarfiles):
        print(f"Processing {i} of {total}", end="\r")
        tar = tarfile.open(filename, "r")

        # This is a tar info
        for member in tar.getmembers():
            jobid = os.path.basename(member.name).replace(".json", "")
            if member.isdir() or not member.name.endswith("json"):
                continue
            content = tar.extractfile(member).read()
            digest = content_hash(content)
            if digest in digests:
                duplicates += 1
                continue

            if not content:
                continue
            content = json.loads(content)
            if "BatchScript" not in content["scontrol"]:
                continue
            script = content["scontrol"]["BatchScript"]
            tokens = tokenize(script)
            fd.write(" ".join(tokens) + "\n")
            meta.write(f"{filename} {jobid}\n")

        tar.close()

    fd.close()
    meta.close()


if __name__ == "__main__":
    main()
