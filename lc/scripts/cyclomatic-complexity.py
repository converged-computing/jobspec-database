#!/usr/bin/env python3


import argparse
import csv
import fnmatch
import hashlib
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import tempfile
from io import StringIO

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

here = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(here)

create_sql = """CREATE TABLE IF NOT EXISTS jobspecs (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
jobid TEXT,
sha256 TEXT,
sha1 TEXT,
ccn NUMBER)
"""

# https://www.sqlite.org/limits.html
insert_query = (
    "INSERT INTO jobspecs(name, jobid, sha256, sha1, ccn) VALUES(?, ?, ?, ?, ?)"
)


def remove_upper_outliers(data):
    """
    Remove upper outliers
    """
    bound = np.percentile(data, [95])
    return [x for x in data if x < bound[0]]


def get_parser():
    parser = argparse.ArgumentParser(description="Cyclometric Complexity Calculator")
    parser.add_argument(
        "input",
        help="Input directory",
        default=os.path.join(root, "raw"),
    )
    parser.add_argument(
        "--db",
        help="Output sqlite database",
        default=os.path.join(root, "data", "cyclomatic-complexity.db"),
    )
    parser.add_argument(
        "--outdir",
        help="Output directory",
        default=os.path.join(root, "data"),
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for bulk inserts (defaults to 1000)",
        default=1000,
        type=int,
    )
    return parser


def get_tmpfile(prefix="", suffix=""):
    """
    Get temporary script file
    """
    tmpdir = tempfile.gettempdir()
    prefix = os.path.join(tmpdir, os.path.basename(prefix))
    fd, tmp_file = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return tmp_file


def content_hash(filename, algorithm="sha256"):
    with open(filename, "rb", buffering=0) as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def calculate_complexity(filepath):
    """
    Use shellmetrics to calculate complexity.

    This returns one line per complexity. We are going to return an
    average across functions for one value, but note we could get partial
    breakdown if desired.
    """
    p = subprocess.Popen(
        ["shellmetrics", "--csv", filepath],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    out, err = p.communicate()
    out = out.decode("utf-8")

    # Use stringio to read the csv into csv parser
    f = StringIO(out)
    reader = csv.reader(f, delimiter=",")
    rows = list(reader)

    # The main is always the second row
    if rows[0][4] != "ccn":
        raise ValueError(f"Unexpected column headers for {filepath}:\n{out}")
    # This is the ccn score for <main> - it is a string parsed to int

    ccns = []
    for row in rows[1:]:
        # These don't seem to be included in the mean ccn in the pretty UI
        if row[1] in ["<begin>", "<end>"]:
            continue
        ccns.append(int(row[4]))

    # This should not happen, but for really simply stuff it seems to.
    # Let's give a value of 0
    if not ccns:
        print(f"Warning: no CCN scores found for {filepath}: assigning value of 0")
        return 0
    return statistics.mean(ccns)


def read_json(input_file):
    """
    Read json from an input file.
    """
    with open(input_file, "r") as filey:
        data = json.loads(filey.read())
    return data


def calculate_digests(filepath):
    """
    Note that we aren't removing duplicates here.
    """
    sha256_digest = content_hash(filepath, "sha256")
    sha1_digest = content_hash(filepath, "sha1")
    return sha256_digest, sha1_digest


def write_file(content, filename):
    """
    Write some text content to a file
    """
    with open(filename, "w") as fd:
        fd.write(content)


def path_to_prefix(indir, path):
    return path.replace(indir + os.sep, "").replace(os.sep, "-").rsplit(".", 1)[0] + "-"


def iter_jobspecs(indir):
    """
    The LC database has a combination of .tar.gz (members)
    and json files, and we need to parse both.
    """
    # First process json files
    files = list(recursive_find(indir, "*.json"))
    total = len(files)
    for i, filename in enumerate(files):
        print(f"Processing {i} of {total} json files", end="\r")
        # Only include those with batch script
        content = read_json(filename)
        if "BatchScript" not in content["scontrol"]:
            continue
        jobid = os.path.basename(filename).replace(".json", "")
        script = content["scontrol"]["BatchScript"]
        tmpfile = get_tmpfile(prefix=path_to_prefix(indir, filename), suffix=".sh")
        write_file(script, tmpfile)

        # file for reading, actual file name, and jobid
        yield tmpfile, filename, jobid

        # Clean up after we use it
        if os.path.exists(tmpfile):
            os.remove(tmpfile)

    # Now read the tars
    tarfiles = list(recursive_find(indir, "*.tar"))
    total = len(tarfiles)
    for i, filename in enumerate(tarfiles):
        print(f"Processing {i} of {total} tarfiles", end="\r")
        tar = tarfile.open(filename, "r")

        # This is a tar info
        for member in tar.getmembers():
            if member.isdir() or not member.name.endswith("json"):
                continue
            prefix = path_to_prefix(indir, filename)
            jobid = os.path.basename(member.name).replace(".json", "")
            content = tar.extractfile(member).read()
            if not content:
                continue
            content = json.loads(content)
            if "BatchScript" not in content["scontrol"]:
                continue
            script = content["scontrol"]["BatchScript"]
            tmpfile = get_tmpfile(prefix=prefix + jobid + "-", suffix=".sh")
            write_file(script, tmpfile)

            # file for reading, actual file name, and jobid
            yield tmpfile, filename, jobid

            # Clean up after we use it
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

        tar.close()


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

    # shellmetrics must be on path!
    if shutil.which("shellmetrics") is None:
        sys.exit(
            "Please install shellmetrics from: https://github.com/shellspec/shellmetrics"
        )

    # Create the table
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()
    cursor.execute(create_sql)

    # Let's do batches of 1K, we can go higher but that's ok :)
    # https://www.sqlite.org/limits.html
    inserts = []
    for parts in iter_jobspecs(args.input):
        filename, name, jobid = parts
        sha256, sha1 = calculate_digests(filename)
        ccn = calculate_complexity(filename)
        # We insert the name of the original path
        # and not the parsed one
        inserts.append((name, jobid, sha256, sha1, ccn))

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

    # Make some plots!
    values = cursor.execute("SELECT ccn from jobspecs;").fetchall()
    values = [x[0] for x in values]
    values.sort()

    # Plot with outliers removed
    without_outliers = remove_upper_outliers(values)
    number_outliers = len(values) - len(without_outliers)
    # There are 7354 upper outliers (above 2)
    print(f"There are {number_outliers} upper outliers")

    plt.figure(figsize=(6, 3))
    sns.histplot(without_outliers, bins=5)
    plt.title("Cyclomatic Complexity for LC JobSpecs")
    plt.savefig(os.path.join(args.outdir, "cyclomatic-complexity.png"))
    plt.clf()

    # Plot the outliers too (above 1.5)
    outliers = [x for x in values if x > np.max(without_outliers)]
    sns.histplot(outliers)
    plt.title("Cyclomatic Complexity for LC JobSpecs (outliers)")
    plt.savefig(os.path.join(args.outdir, "cyclomatic-complexity-outliers.png"))
    plt.clf()

    cursor.close()


if __name__ == "__main__":
    main()
