#!/usr/bin/env/python

import sys
import os
import re
import json
import fnmatch
import argparse


here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)


def read_file(filename):
    with open(filename, "r") as fd:
        content = fd.read()
    return content


def write_json(obj, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(obj, indent=4))


def read_json(filename):
    with open(filename, "r") as fd:
        content = json.loads(fd.read())
    return content


def get_parser():
    parser = argparse.ArgumentParser(description="mistral with template")
    parser.add_argument(
        "--input",
        help="Input directory",
        default=os.path.join(here, "data", "gemini-with-template"),
    )
    parser.add_argument(
        "--original",
        help="Original data directory (has same path)",
        default=os.path.join(root, "data"),
    )
    parser.add_argument(
        "--output",
        help="Output data directory",
        default=os.path.join(here, "data", "mistral-with-template-processed"),
    )
    return parser


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def attempt_to_int(res, key, value):
    try:
        res[key] = int(value)
    except:
        pass


def parse_timestamp(value):
    """
    Parse a timestamp string to an integer time in seconds.
    """
    seconds = 0
    if "-" in value:
        days, value = value.split("-", 1)
        seconds += int(days) * (24 * 60 * 60)
    if value.count(":") == 0:
        seconds += int(value)
    elif value.count(":") == 3:
        days, hours, minutes, secs = value.split(":")
        seconds += (
            int(days) * (24 * 60 * 60)
            + int(secs)
            + (int(minutes) * 60)
            + (int(hours) * 60 * 60)
        )
    elif value.count(":") == 1:
        minutes, secs = value.split(":")
        seconds += int(secs) + (int(minutes) * 60)
    elif value.count(":") == 2:
        hours, minutes, secs = value.split(":")
        seconds += int(secs) + (int(minutes) * 60) + (int(hours) * 60 * 60)
    else:
        return None
    return seconds


def read_manual_resources():
    """
    Read in (and normalize) manual resources.
    """
    manual = json.loads(
        read_file(os.path.join(here, "data", "combined", "jobspec-directives.json"))
    )

    # Let's parse our resources into consistent sets, will full paths
    # Try to normalize into one format. We do try/except because some are templates
    # with strings, and arguably we just want things we can parse
    resources = {}
    for path, r in manual.items():
        path = os.path.abspath(path)
        job_r = {}

        # Create keys and values to pop
        items = []

        # Ensure we process all of them
        for key, value in r.items():
            items.append([key, value])

        # Ensure we process all of them
        while items:
            key, value = items.pop(0)

            if key.startswith("l "):
                key = key.replace("l ", "")

            if key in ["t", "time", "walltime", "W", "lwalltime"]:
                # Try to normalize time
                if not isinstance(value, str) or re.search(
                    "([$]|[>]|[<][}][{])", value
                ):
                    continue
                try:
                    value = parse_timestamp(value.split(",")[0])
                except:
                    pass
                if value is not None:
                    job_r["time"] = value

            elif key in ["mem", "pmem"]:
                job_r["memory"] = value

            # ignore these
            elif key in [
                "o",
                "e",
                "output",
                "error",
                "p",
                "partition",
                "comment",
                "P",
                "job-name",
                "q",
                "cwd",
            ]:
                continue
            elif key == "exclusive":
                job_r["exclusive"] = value

            elif key in ["N", "nodes", "nnodes"]:
                attempt_to_int(job_r, "nodes", value)

            elif key in ["g", "ngpus", "gpus"]:
                attempt_to_int(job_r, "gpus", value)

            # Are these the same?
            elif key in ["n", "ntasks", "c", "ncpus", "mpiprocs"]:
                attempt_to_int(job_r, "tasks", value)

            elif key == "tasks-per-node":
                attempt_to_int(job_r, "ntasks-per-node", value)

            elif key in ["ntasks-per-core", "threads-per-core"]:
                attempt_to_int(job_r, "ntasks-per-core", value)

            # Take these as is
            elif key in [
                "cpus-per-task",
                "ntasks-per-node",
                "gpu_type",
                "gres",
                "ntasks",
                "gpus-per-node",
                "mem-per-cpu",
                "gres-flags",
                "ntasks-per-socket",
                "cpus-per-gpu",
                "gpus-per-task",
                "cores-per-socket",
                "sockets-per-node",
                "mem-per-gpu",
            ]:
                try:
                    job_r[key] = int(value)
                except:
                    job_r[key] = value

            elif key in ["ppn"]:
                attempt_to_int(job_r, "ntasks-per-node", value)

            elif isinstance(value, str) and "=" in value and ":" in value:
                sets = [x for x in value.split(":") if "=" in x]
                for item in sets:
                    k, v = item.split("=", 1)
                    items.append([k, v])

        resources[path] = job_r
    return resources


def normalize_gemini(gemini_set):
    """
    Given gemini resources, parse to int and normalize and remove
    empty values like we did for manual parsing. This will make it possible
    to compare between the two.
    """
    parsed = {}
    for key, value in gemini_set.items():
        if value in ["", None]:
            continue

        # These should parse into integers
        if key in [
            "cpus_per_task",
            "tasks",
            "gpus",
            "gpus_per_node",
            "cores_per_socket",
            "gpus_per_task",
            "cpus_per_gpu",
            "ntasks_per_node",
            "nodes",
            "ntasks_per_core",
            "sockets_per_node",
            "ntasks_per_socket",
        ]:
            # I could try custom parsing here, but likely we can't do that
            # we need to allow gemini to fail to get something right. E.g., 1:ppn=12
            # is not fully parsed
            attempt_to_int(parsed, key, value)

        elif key == "time":
            if not isinstance(value, str) or re.search("([$]|[>]|[<][}][{])", value):
                continue
            try:
                value = parse_timestamp(value.split(",")[0])
            except:
                pass
            if value is not None:
                parsed["time"] = value
        else:
            parsed[key] = value
    return parsed


prompt = "Can you tell me what application this script is running and put the answer as a single term on the first line, followed by more detail about the other software and resource requirements in the script? And can you give me an output format in raw json\n"


def get_app(result):
    """
    Shared function to get an application
    """
    # This is consistent
    app = None
    if "Application" in result:
        app = result["Application"].lower()
    elif "application" in result:
        app = result["application"].lower()
    return app


def chunkize(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input):
        sys.exit("An input directory is required.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read in the output json files (and compare to raw results)
    total = len(list(recursive_find(args.input, "*-response.json")))

    # There are a total of 33239 Gemini API query results
    # We are using these since they get us started with the actual processing
    print(f"There are a total of {total} parsed jobspec queries")

    # We have 32668 parsed JSON results
    contenders = list(recursive_find(args.input, "*-answer.json"))
    print(f"We have {len(contenders)} parsed JSON results")

    # These are my manual (regular expression) parsing of resources
    resources = read_manual_resources()

    # Here are unique resource - this is a starting set, probably not perfect
    resource_names = set()
    for _, r in resources.items():
        for key in r:
            resource_names.add(key)

    # For each answer (which has parsed resources, etc) assemble the mistral AI
    # expected training format.
    instruct_outdir = os.path.join(args.output, "instruct")
    pretrain_outdir = os.path.join(args.output, "pretrain")
    for dirname in instruct_outdir, pretrain_outdir:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # https://github.com/mistralai/mistral-finetune/
    for i, chunk in enumerate(chunkize(contenders, 10)):
        if i > 10:
            break
        output_file = os.path.join(pretrain_outdir, f"mistral-training-chunk-{i}.jsonl")
        output_pre_file = os.path.join(
            instruct_outdir, f"mistral-pre-training-chunk-{i}.jsonl"
        )

        # This is a json LINES file, weird
        pout = open(output_pre_file, "w")
        iout = open(output_file, "w")

        for contender in chunk:
            answer = read_json(contender)

            # This is the original content
            original = read_file(
                os.path.join(
                    args.original, contender.replace(args.input + os.sep, "")
                ).replace("-answer.json", "")
            )
            json.dump({"text": original}, pout)
            pout.write("\n")
            messages = {
                "messages": [
                    {"role": "user", "content": prompt + original},
                    {"role": "assistant", "content": answer},
                ]
            }
            json.dump(messages, iout)
            iout.write("\n")

        pout.close()
        iout.close()


if __name__ == "__main__":
    main()
