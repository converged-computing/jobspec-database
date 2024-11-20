#!/usr/bin/env python3


import numpy as np
import argparse
import fnmatch
import hashlib
import os
import statistics
import shutil
import sys
import sqlite3
import subprocess
import seaborn as sns
import matplotlib.pylab as plt

from io import StringIO
import csv

here = os.path.abspath(os.path.dirname(__file__))

create_sql = """CREATE TABLE IF NOT EXISTS jobspecs (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,
sha256 TEXT,
sha1 TEXT,
ccn NUMBER)
"""

# https://www.sqlite.org/limits.html
insert_query = "INSERT INTO jobspecs(name, sha256, sha1, ccn) VALUES(?, ?, ?, ?)"


def remove_upper_outliers(data):
    """
    Remove upper outliers
    """
    bound = np.percentile(data, [95])
    return [x for x in data if x < bound[0]]


def get_parser():
    parser = argparse.ArgumentParser(description="Cyclomatic Complexity Calculator")
    parser.add_argument(
        "input",
        help="Input directory",
        default=os.path.join(here, "data"),
    )
    parser.add_argument(
        "--db",
        help="Output sqlite database",
        default=os.path.join(here, "data", "cyclomatic-complexity.db"),
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


def calculate_digests(filepath):
    """
    Note that we aren't removing duplicates here.
    """
    sha256_digest = content_hash(filepath, "sha256")
    sha1_digest = content_hash(filepath, "sha1")
    return sha256_digest, sha1_digest


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
        sha256, sha1 = calculate_digests(filename)
        ccn = calculate_complexity(filename)
        inserts.append((filename, sha256, sha1, ccn))

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
