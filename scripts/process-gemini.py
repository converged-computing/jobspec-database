#!/usr/bin/env/python

import seaborn as sns
import matplotlib.pylab as plt
import pandas
import numpy
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


def get_parser():
    parser = argparse.ArgumentParser(description="gemini with template")
    parser.add_argument(
        "--input",
        help="Input directory",
        default=os.path.join(here, "data", "gemini-with-template"),
    )
    parser.add_argument(
        "--output",
        help="Output data directory",
        default=os.path.join(here, "data", "gemini-with-template-processed"),
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

    # There are a total of 33239 Gemini API queries
    print(f"There are a total of {total} Gemini API queries")

    # We have 32668 parsed JSON results
    # A lot of these would parse if we re-ran. In many cases there is
    # some illegal character that prevents saving to json 9and we have the result)
    contenders = list(recursive_find(args.input, "*-answer.json"))
    print(f"We have {len(contenders)} parsed JSON results")

    # These are my manual (regular expression) parsing of resources
    resources = read_manual_resources()

    # Here are unique resource - this is a starting set, probably not perfect
    resource_names = set()
    for _, r in resources.items():
        for key in r:
            resource_names.add(key)

    # Create a data frame
    # Note we will need to custom parse memory too
    columns = list(resource_names)

    # 1. Looking at resources across applications.
    # Here we are going to put our resources into a matrix, and then
    # make plots, one per type, across applications that Gemini has given
    # us.

    # 2. We are also going to save the resources that Gemini has
    # predicted and do a similar comparison to what we manually
    # parsed. This will tell us how good the model is, at least
    # at that part, but we might extend that trust to use
    # the rest it gives us.
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

    # TODO - parse this into sections
    # see how much my resource parsing agrees with this
    # look at breakdown unique applications
    # look for associations between applications / resources
    # First create set of unique applications, and also software included
    apps = {}
    software = {}

    # Find meaningful applications first, and associate the
    # parsed jobspecs with what gemini predicts for resourcesresour
    gemini_resources = {}
    for i, filename in enumerate(contenders):
        result = json.loads(read_file(filename))
        original_filename = filename.replace("-answer.json", "")
        original_filename = original_filename.replace(
            "scripts/data/gemini-with-template", "data"
        )
        gemini_resources[original_filename] = result["resources"]
        software[original_filename] = list(
            set([result["application"]] + result["software"])
        )
        app = get_app(result)
        if app is not None:
            if app not in apps:
                apps[app] = 0
            apps[app] += 1

    # Make a histogram of counts
    apps = {
        k: v for k, v in sorted(apps.items(), reverse=True, key=lambda item: item[1])
    }

    # Save original lists of software, apps, and resources
    write_json(resources, os.path.join(args.output, "resources-manual-parsing.json"))
    write_json(gemini_resources, os.path.join(args.output, "gemini-resources.json"))
    write_json(software, os.path.join(args.output, "gemini-software.json"))

    # Let's cut at 32 - Russ said it's statistically important for sampling
    # we get close to the normal distribution. I'm going to custom filter these to
    # a set I know are applications (e.g., not slurm)
    skips = ["slurm", "pbs", "lsf"]
    filtered = {k: v for k, v in apps.items() if v >= 32 and k not in skips}

    # Now let's filter our matrix and data to apps in this list
    # This is going to help us to make plots that show our parsed
    # resources against the application sets determined by Gemini
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
        basename = filename.split(os.sep + "gemini-with-template" + os.sep, 1)[
            -1
        ].strip(os.sep)
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
    matrix_filtered.to_csv(
        os.path.join(args.output, "gemini-applications-with-manual-resources.csv")
    )

    # Can skip this since we already did it, and takes a while!
    # plot_manual_resources_against_apps(matrix_filtered, columns, args)

    # 2. Now we go back to bullet 2 - comparing our resources with Gemini.
    # We first need to map between the two. The set we are using is in "columns"
    scores = {}
    parsed_count = 0

    # Prepare file for people to look at
    visuals = {}
    visuals_with_wrong = {}
    visuals_with_missing = {}

    # Which fields did we get wrong?
    fields_wrong = {}
    fields_missing = {}

    for filename, resource_set in resources.items():
        gemini_set = gemini_resources.get(filename)
        if not gemini_set:
            continue
        parsed_count += 1

        basename = os.path.basename(filename)
        content = read_file(filename)

        # Save this for printing
        visual = f"""#{basename}

```bash
{content}
```

## Manual vs Gemini

```python
Manual: {resource_set}
Gemini: {gemini_set}
```
"""
        visuals[filename] = visual

        # Gemini can add extra back that we filtered out - ignore
        resource_set = {k.replace("-", "_"): v for k, v in resource_set.items()}

        # This was a typo - likely we can't use this value to assess
        for item in [gemini_set, resource_set]:
            if "ntasks_per_code" in item:
                item["ntasks_per_core"] = item["ntasks_per_code"]
                del item["ntasks_per_code"]

        # Next parse units for memory, time, for gemini
        gemini_parsed = normalize_gemini(gemini_set)

        # This is us improving our manually parsed set to convert gres to gpu
        # Note that I don't know what a gpu_cluster:6 means
        if "gres" in resource_set and resource_set["gres"].startswith("gpu"):
            resource_set["gpu"] = resource_set["gres"].split("gpu:", 1)[-1]
            rest = resource_set["gres"].replace("gpu:" + resource_set["gpu"], "")
            if rest and "gpumem:" in rest:
                try:
                    resource_set["mem_per_gpu"] = int(rest.replace("gpumem:", ""))
                except:
                    resource_set["mem_per_gpu"] = rest.replace("gpumem:", "")
            # This can be a number OR A100:1 etc
            try:
                resource_set["gpu"] = int(resource_set["gres"].split("gpu:", 1)[-1])
            except:
                pass

        print()
        print(f"Manual: {resource_set}")
        print(f"Gemini: {gemini_parsed}")

        # Of the ones that I have, count the number that are present, and the number that
        # are correct or wrong exactly. Also keep track of missing
        score = {
            "missing": 0,
            "correct": 0,
            "present_and_wrong": 0,
            "total": len(resource_set),
        }
        for key, value in resource_set.items():
            if key not in gemini_parsed:
                score["missing"] += 1
                visuals_with_missing[filename] = visual

                # Keep track of which ones missing
                if key not in fields_missing:
                    fields_missing[key] = 0
                fields_missing[key] += 1
                continue

            # If the key is memory, try to convert both to integer
            if key == "memory":
                try:
                    value = int(value)
                    gemini_parsed[key] = value
                except:
                    pass
            if key in gemini_parsed and gemini_parsed[key] == value:
                score["correct"] += 1
            else:
                score["present_and_wrong"] += 1
                visuals_with_wrong[filename] = visual
                if key not in fields_wrong:
                    fields_wrong[key] = set()
                fields_wrong[key].add((value, gemini_parsed[key]))
            scores[filename] = score

    fields_missing = {
        k: v
        for k, v in sorted(
            fields_missing.items(), key=lambda item: item[1], reverse=True
        )
    }

    # Convert to list to save
    fields_wrong = {k: list(v) for k, v in fields_wrong.items()}

    # We were able to compare 32656 resource sets between Gemini and manual parsing
    print(
        f"We were able to compare {parsed_count} resource sets between Gemini and manual parsing"
    )
    write_json(scores, os.path.join(args.output, "single-scores.json"))
    write_json(visuals, os.path.join(args.output, "visual-resource-comparison.json"))
    write_json(
        visuals_with_missing,
        os.path.join(args.output, "visual-with-missing-resource-comparison.json"),
    )
    write_json(
        visuals_with_wrong,
        os.path.join(args.output, "visual-with-wrong-resource-comparison.json"),
    )
    write_json(
        fields_missing,
        os.path.join(args.output, "fields-missing-counts-resource-comparison.json"),
    )
    write_json(
        fields_wrong,
        os.path.join(args.output, "fields-wrong-visual-resource-comparison.json"),
    )

    # Calculate the total scores
    # I didn't use this, didn't plot them
    # df = pandas.DataFrame(columns=['metric', 'value'])
    # idx = 0
    # for _, score in scores.items():
    #    for key, value in score.items():
    #        if key == "total":
    #            continue
    #        df.loc[idx, :] = [key, value]
    #        idx += 1
    # df.to_csv(os.path.join(args.output, 'scores-summary-df.csv'))

    # Look at totals
    totals = {"correct": 0, "missing": 0, "present_and_wrong": 0, "total": 0}
    for _, score in scores.items():
        for key, value in score.items():
            totals[key] += value

    # Totals
    # {'correct': 90102,
    # 'missing': 13389,
    # 'present_and_wrong': 6218,
    # 'total': 109709}


def plot_manual_resources_against_apps(matrix_filtered, columns, args):
    """
    Plot the manual resources we parsed against Gemini application set
    """
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
