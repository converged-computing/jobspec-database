#!/usr/bin/env python3


import numpy as np
import argparse
import fnmatch
import hashlib
import os
import statistics
import shutil
import re
import sys
import sqlite3
import subprocess
import seaborn as sns
import json
import pandas
import matplotlib.pylab as plt

from io import StringIO
import csv

here = os.path.abspath(os.path.dirname(__file__))

# tags is a json list
create_sql = """CREATE TABLE IF NOT EXISTS jobspecs (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
manager_tags TEXT,
software_tags TEXT,
software_tags_with_template TEXT,
length NUMBER)
"""

# https://www.sqlite.org/limits.html
insert_query = "INSERT INTO jobspecs(name, manager_tags, software_tags, software_tags_with_template, length) VALUES(?, ?, ?, ?, ?)"


# Software from gemini "in the wild"
software = None
counts = None

# Software with a template for gemini
software_with_template = None
counts_with_template = None
skipped = set()


def get_parser():
    parser = argparse.ArgumentParser(description="Summarize Jobspecs")
    parser.add_argument(
        "input",
        help="Input directory",
        default=os.path.join(here, "data"),
    )
    parser.add_argument(
        "--db",
        help="Output sqlite database",
        default=os.path.join(here, "data", "jobspec-summary-github.db"),
    )
    parser.add_argument(
        "--outdir",
        help="Output directory",
        default=os.path.join(here, "data"),
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for bulk inserts (defaults to 1000)",
        default=1000,
        type=int,
    )
    return parser


def read_file(filename):
    with open(filename, "r") as f:
        content = f.read()
    return content


def write_json(obj, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(obj, indent=4))


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def find_tags(filename):
    """
    Given a jobspec file, look for job directive tags
    """
    global software
    global software_with_template
    global counts
    global counts_with_template
    global skipped

    # These are the directives we searched for in search.py
    search_to_manager = {
        "#FLUX": "flux",
        "#SBATCH": "slurm",
        "#PBATCH": "pbs",
        "#COBALT": "cobalt",
        "#PBS": "pbs",
        "#OAR": "oar",
        "#BSUB": "lsf",
    }
    regex = "(%s)" % "|".join(list(search_to_manager.keys()))
    content = read_file(filename)
    managers = [search_to_manager[x] for x in set(re.findall(regex, content))]

    # Too risky to use regex
    apps = [x for x in software if x in content]
    apps_template = [x for x in software_with_template if x in content]

    for app in apps:
        counts[app] += 1
    for app in apps_template:
        counts_with_template[app] += 1
    return json.dumps(apps), json.dumps(apps_template), json.dumps(managers)


def clean_software_set(software):
    """
    This is parsed from gemini (without a template) so clean it up a bit.
    """
    # Take a basename, make all lowercase
    software = list(set([os.path.basename(x).lower() for x in software]))

    # Remove everything except for .
    software = [re.sub("[^A-Za-z0-9.]+", "", x) for x in software]

    # Assume we don't want python or shell scripts, or singularity
    software = [x for x in software if not re.search("[.](sh|py|sif)", x)]

    # No empty / None / 0 allowed
    software = [x for x in software if x]

    # Greater than length of 1
    software = [x for x in software if len(x) > 1]

    updated = []
    for x in software:
        try:
            float(x)
            pass
        except:
            updated.append(x)
    return updated


def main():
    """
    jobspec feature parsing
    """
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input):
        sys.exit("An input directory is required.")

    if os.path.exists(args.db):
        sys.exit(
            f"Database file {args.db} already exists - remove or provide a different path."
        )

    # Read in the gemini softwre list (exhaustive)
    global software
    global software_with_template
    software = json.loads(
        read_file(
            os.path.join(
                here,
                "data",
                "gemini-with-template-processed",
                "gemini-software-list.json",
            )
        )
    )
    # For the plot, let's use the gemini filtered (processed) data
    df = pandas.read_csv(
        os.path.join(
            here,
            "data",
            "gemini-with-template-processed",
            "gemini-applications-with-manual-resources.csv",
        )
    )
    software_with_template = list(df.application.unique())

    # Save a list to look at later
    outdir = os.path.join(here, "data", "jobspec-summary")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get rid of numbers, etc
    software = clean_software_set(software)

    # Save to file for human inspection
    summary_list = os.path.join(outdir, "software-list.json")
    write_json(software, summary_list)

    # Create a lookup of counts for each
    global counts
    global counts_with_template
    counts = {k: 0 for k in software}
    counts_with_template = {k: 0 for k in software_with_template}

    # Create the table
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    cursor.execute(create_sql)

    # Read in text, and as we go, generate content hash.
    # We don't want to use duplicates (forks are disabled, but just being careful)
    files = list(recursive_find(args.input))

    # Let's do batches of 1K, we can go higher but that's ok :)
    # https://www.sqlite.org/limits.html
    inserts = []
    total = len(files)
    for i, filename in enumerate(files):
        # Skip jobspec associated files
        print(f"{i}/{total}", end="\r")
        if "jobspec-cfg" in filename:
            continue

        apps, apps_with_template, managers = find_tags(filename)
        content = read_file(filename)
        inserts.append((filename, managers, apps, apps_with_template, len(content)))

        # Insert and reset
        if len(inserts) >= args.batch_size:
            print(f"\nInserting {len(inserts)} into database.")
            cursor.executemany(insert_query, inserts)
            conn.commit()
            inserts = []

    # Last one? Probably.
    if inserts:
        cursor.executemany(insert_query, inserts)
        conn.commit()

    import IPython

    IPython.embed()
    sys.exit()

    # Save app counts to file after sorting
    counts = {
        k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    }
    counts_subset = {k: v for k, v in counts.items() if v >= 32}
    counts_file = os.path.join(outdir, "software-filtered-counts.json")
    write_json(counts_subset, counts_file)

    # These are the counts of applications from the gemini processed with template set
    counts_file = os.path.join(outdir, "gemini-with-template-apps-counts.json")
    counts_with_template = {
        k: v
        for k, v in sorted(
            counts_with_template.items(), key=lambda item: item[1], reverse=True
        )
    }
    write_json(counts_with_template, counts_file)

    count_df = pandas.DataFrame(df.application.value_counts())
    count_df["application"] = count_df.index.tolist()

    # Reset index to numbers
    count_df.index = list(range(count_df.shape[0]))

    # Plot application counts
    fig, ax = plt.subplots(figsize=(20, 8))
    ax = sns.scatterplot(
        data=count_df,
        x="application",
        y="count",
        palette="Set3",
    )
    plt.title(f"Application Counts >= 32 JobSpecs")
    ax.set_xlabel("application", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticks(), fontsize=8)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    plt.legend([], [], frameon=False)
    plt.savefig(os.path.join(outdir, f"software-counts.png"))
    plt.clf()
    plt.close()

    # Make another derivative with the templated counts
    count_df["source"] = "gemini-classification"
    idx = count_df.shape[0]
    for app, count in counts_with_template.items():
        count_df.loc[idx, :] = [count, app, "manual-search"]
        idx += 1

    fig, ax = plt.subplots(figsize=(20, 8))
    ax = sns.scatterplot(
        data=count_df,
        x="application",
        y="count",
        palette="Set3",
        hue="source",
    )
    plt.title(f"Application Counts >= 32 JobSpecs")
    ax.set_xlabel("application", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticks(), fontsize=8)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    plt.legend([], [], frameon=False)
    plt.savefig(os.path.join(outdir, f"software-counts-manual-vs-gemini.png"))
    plt.clf()
    plt.close()

    # Get all the workload managers for each so we can derive counts
    values = cursor.execute("SELECT manager_tags from jobspecs;").fetchall()
    values = [json.loads(x[0]) for x in values]
    manager_counts = {}
    for listing in values:
        for value in listing:
            if value not in manager_counts:
                manager_counts[value] = 0
            manager_counts[value] += 1

    manager_counts = {
        k: v
        for k, v in sorted(
            manager_counts.items(), key=lambda item: item[1], reverse=True
        )
    }
    print(json.dumps(manager_counts, indent=4))

    # Finally, the lengths
    values = cursor.execute("SELECT length from jobspecs;").fetchall()
    values = [x[0] for x in values]

    # Make some plots! Let's choose counts >= 32
    plt.figure(figsize=(20, 3))
    sns.histplot(values, bins=100)
    total = len(values)
    plt.title(f"JobSpecs Lengths (N={total})")
    plt.savefig(os.path.join(outdir, "jobspec-github-lengths.png"))
    plt.clf()

    how_many_above_50k = len([x for x in values if x > 50000])
    print(
        f"There are {how_many_above_50k} scripts longer than 50K characters (including white space)"
    )
    values = [x for x in values if x < 50000]
    plt.figure(figsize=(20, 3))
    sns.histplot(values, bins=100)
    total = len(values)
    plt.title(f"JobSpecs Lengths (N={total})")
    plt.savefig(os.path.join(outdir, "jobspec-github-lengths-lt-50k.png"))
    plt.clf()

    # Plot with outliers removed
    without_outliers = remove_upper_outliers(values)
    number_outliers = len(values) - len(without_outliers)
    # Above a value of 8
    print(f"There are {number_outliers} upper outliers")

    plt.figure(figsize=(6, 3))
    sns.histplot(without_outliers, bins=8)
    plt.title("Cyclomatic Complexity for JobSpecs")
    plt.savefig(os.path.join(args.outdir, "cyclomatic-complexity.png"))
    plt.clf()

    # Plot the outliers too
    max_value = np.max(without_outliers)
    outliers = [x for x in values if x > max_value]
    sns.histplot(outliers)
    plt.title("Cyclomatic Complexity for JobSpecs (outliers)")
    plt.savefig(os.path.join(args.outdir, "cyclomatic-complexity-outliers.png"))
    plt.clf()
    cursor.close()


if __name__ == "__main__":
    main()
