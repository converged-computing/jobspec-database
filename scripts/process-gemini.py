#!/usr/bin/env/python

import seaborn as sns
import matplotlib.pylab as plt
import pandas
import numpy
import sys
import os
import re
import time
import json
import hashlib
import fnmatch
import argparse
import pathlib
import textwrap
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


def get_parser():
    parser = argparse.ArgumentParser(description="word2vec")
    parser.add_argument(
        "--input",
        help="Input directory",
        default=os.path.join(here, "data", "gemini"),
    )
    parser.add_argument(
        "--output",
        help="Output data directory",
        default=os.path.join(here, "data", "gemini-processed"),
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


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input):
        sys.exit("An input directory is required.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read in the output json files (and compare to raw results)
    total = len(list(recursive_find(args.input, "*-response.json")))

    # There are a total of 32610 Gemini API queries
    print(f"There are a total of {total} Gemini API queries")

    # We have 30779 parsed JSON results
    # A lot of these would parse if we re-ran. In many cases there is
    # some illegal character that prevents saving to json 9and we have the result)
    contenders = list(recursive_find(args.input, "*-answer.json"))
    print(f"We have {len(contenders)} parsed JSON results")

    # First create set of unique applications, and also software included
    apps = {}
    software = set()

    # These are my manual (regular expression) parsing of resources
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
                if not isinstance(value, str) or re.search("([$]|[>]|[<])", value):
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
                attempt_to_int(job_r, "ntasks-per-code", value)

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

    # Here are unique resource - this is a starting set, probably not perfect
    resource_names = set()
    for _, r in resources.items():
        for key in r:
            resource_names.add(key)

    # Create a data frame
    # Note we will need to custom parse memory too
    columns = list(resource_names)
    import IPython

    IPython.embed()
    sys.exit()

    matrix = numpy.zeros((len(resources), len(columns)))

    # Also keep a lookup of the filename index
    lookup = {}
    for row, (filename, r) in enumerate(resources.items()):
        # Make the filename relative to the repository root
        basename = filename.split(os.sep + "data" + os.sep, 1)[-1].strip(os.sep)
        lookup[basename] = row
        values = []
        # This doesn't parse time... they are all different formats...
        for column in columns:
            if column in r and isinstance(r[column], int):
                values.append(r[column])
            else:
                values.append(0)
        matrix[row] = values

    # Observations
    # We need to give it an output template with what we want (not consistent)
    # With this setup I can't easily compare resource parsing

    # TODO - parse this into sections
    # see how much my resource parsing agrees with this
    # look at breakdown unique applications
    # look for associations between applications / resources

    # Let's first find some meaningful set of applications
    # "SLURM" is not it :)
    for i, filename in enumerate(contenders):
        result = json.loads(read_file(filename))

        app = get_app(result)
        if app is not None:
            if app not in apps:
                apps[app] = 0
            apps[app] += 1

    # Make a histogram of counts
    apps = {
        k: v for k, v in sorted(apps.items(), reverse=True, key=lambda item: item[1])
    }

    # Let's cut at 32 - Russ said it's statistically important for sampling
    # we get close to the normal distribution. I'm going to custom filter these to
    # a set I know are applications (e.g., not slurm)
    skips = ["slurm", "pbs", "lsf"]
    filtered = {k: v for k, v in apps.items() if v >= 32 and k not in skips}

    # Now let's filter our matrix and data to apps in this list
    matrix_filtered = []
    lookup_updated = {}
    new_index = []
    apps = []
    for i, filename in enumerate(contenders):
        result = json.loads(read_file(filename))
        app = get_app(result)
        if app is None or app not in filtered:
            continue
        apps.append(app)
        # lookup the filename in the matrix
        filename = filename.replace("-answer.json", "")
        basename = filename.split(os.sep + "gemini" + os.sep, 1)[-1].strip(os.sep)
        idx = lookup[basename]
        matrix_filtered.append(matrix[idx])
        lookup_updated[basename] = i
        new_index.append(basename)

    # Meh, make pandas. They certainly don't want to.
    matrix_filtered = pandas.DataFrame(numpy.array(matrix_filtered))
    matrix_filtered.columns = columns
    matrix_filtered["application"] = apps
    matrix_filtered.index = new_index

    print(f"Filtered matrix to relevant applications is size {matrix_filtered.shape}")

    # For now remove this outlier - it's huge and not representative
    # WHO RUNS ON 15974 NODES, WHAT KIND OF MONSTER ARE YOU
    # and where can I get a cluster like that too? :D
    matrix_filtered = matrix_filtered[
        matrix_filtered.nodes != matrix_filtered.nodes.max()
    ]

    # IMPORTANT - I'm removing the two jobs that are outliers.
    # We probably shouldn't do this but I want to see the rest of the plot
    # Same with this toon.
    matrix_filtered = matrix_filtered[
        matrix_filtered.time != matrix_filtered.time.max()
    ]
    matrix_filtered = matrix_filtered[
        matrix_filtered.time != matrix_filtered.time.max()
    ]

    # Visualize data - make plot on level of column
    for c, column in enumerate(columns):
        fig, ax = plt.subplots(figsize=(20, 8))
        # Plot lammps times
        df = matrix_filtered.sort_values(column)
        ax = sns.scatterplot(
            data=df,
            x="application",
            y=column,
            hue="application",
            palette="Set3",
        )
        plt.title(f"Resource {column} across applications")
        ax.set_xlabel("application", fontsize=16)
        ax.set_ylabel(column, fontsize=16)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=10)
        ax.set_yticklabels(ax.get_yticks(), fontsize=10)
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.2)
        plt.legend([], [], frameon=False)
        plt.savefig(os.path.join(args.output, f"{column}-resource.png"))
        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
