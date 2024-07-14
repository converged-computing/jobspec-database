#!/usr/bin/env python3

#### PYTHON IMPORTS ################################################################################
import csv
import multiprocessing as mproc
import numpy as np
import os
import re
import shutil
import spacy
import sys
from pathlib import Path


#### CONFIG ########################################################################################
csv.field_size_limit(sys.maxsize)
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    NLP = spacy.load("en")


#### GLOBALS #######################################################################################
ISSUES_HEADER = ["REPO_URL", "REPO_NAME", "REPO_OWNER", "ISSUE_NUMBER", "ISSUE_CREATION_DATE",
    "ISSUE_AUTHOR", "ISSUE_TITLE", "ISSUE_URL", "ISSUE_TEXT", "COMMENT_CREATION_DATE",
    "COMMENT_AUTHOR", "COMMENT_URL", "COMMENT_TEXT"]
COMMITS_HEADER = ["REPO_URL", "REPO_NAME", "REPO_OWNER", "COMMIT_OID", "COMMIT_CREATION_DATE",
    "COMMIT_AUTHOR", "COMMIT_ADDITIONS", "COMMIT_DELETIONS", "COMMIT_HEADLINE", "COMMIT_URL",
    "COMMIT_TEXT", "COMMENT_CREATION_DATE", "COMMENT_AUTHOR", "COMMENT_URL", "COMMENT_TEXT"]
PULL_REQUESTS_HEADER = ["REPO_URL", "REPO_NAME", "REPO_OWNER", "PULL_REQUEST_NUMBER",
    "PULL_REQUEST_TITLE", "PULL_REQUEST_AUTHOR", "PULL_REQUEST_CREATION_DATE", "PULL_REQUEST_URL",
    "PULL_REQUEST_TEXT", "COMMENT_CREATION_DATE", "COMMENT_AUTHOR", "COMMENT_URL", "COMMENT_TEXT"]

RE_PUNCT = re.compile(r"[\.,:;?!]")
RE_WHITESPACE = re.compile(r"[\n\r\t\v\f]") # everything but regular spaces
RE_DUPLICATE_SPACES = re.compile(r"[\s]+")

APOLOGY_LEMMAS = [
    "apology", "apologise", "apologize", "blame", "excuse", "fault", "forgive", "mistake",
    "mistaken", "oops", "pardon", "regret", "sorry"
]


#### FUNCTIONS #####################################################################################
def fixNullBytes(file_pointer):
    """
    Replaces null bytes with "<NULL>" so csv.reader() doesn't produce an error.

    GIVEN:
      file_pointer (...) -- pointer to an open file
    """
    for line in file_pointer:
        yield line.replace("\0", "<NULL>")


def getDataFilepaths(data_dir):
    """
    Get filepaths for issues, commits, and pull requests in the given data_dir.

    GIVEN:
      data_dir (str) -- absolute path to data

    RETURN:
      issues_file (str) -- absolute path to issues CSV
      commits_file (str) -- absolute path to commits CSV
      pull_requests_file (str) -- absolute path to pull requests CSV
    """

    issues_file = os.path.join(data_dir, "issues/issues.csv")
    commits_file = os.path.join(data_dir, "commits/commits.csv")
    pull_requests_file = os.path.join(data_dir, "pull_requests/pull_requests.csv")

    return [issues_file, commits_file, pull_requests_file]


def canonicalize(top_dir):
    """
    Canonicalize filepath.

    GIVEN:
      top_dir (str) -- relative filepath

    RETURN:
      _ (str) -- absolute filepath for top_dir
    """
    return os.path.realpath(top_dir)


def getSubDirNames(top_dir):
    """
    Get the names of subdirectories, one level deep.
    """
    return [d[1] for d in os.walk(top_dir)][0]


def _getRawComments(old_file):
    """
    Retrieve the final column (containing comments) from a CSV file.

    GIVEN:
      old_file (str) -- absolute path to CSV containing data

    RETURN:
      comments (list) -- raw comment text
    """
    comments = list()
    with open(old_file, "r") as f:
        csv_reader = csv.reader(fixNullBytes(f), delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        next(csv_reader)
        for line in csv_reader:
            comments.append(line[-1])

    return comments


def _cleanText(comment):
    """
    Clean up text by lowercaing, removing punctuation, and removing duplicate whitespace.

    GIVEN:
      comment (str) -- a single comment

    RETURN:
      clean_comment (str) -- a cleaned up comment
    """
    clean_comment = RE_PUNCT.sub("", comment.lower())
    clean_comment = RE_WHITESPACE.sub(" ", clean_comment)
    clean_comment = RE_DUPLICATE_SPACES.sub(" ", clean_comment)

    return clean_comment


def _lemmatize(comment):
    """
    Lemmatize a comment.

    GIVEN:
      comment (str) -- a single comment

    RETURN:
      lemmatized_comment (str) -- a lemmatized comment
    """
    comment = NLP(str(comment))
    lemmas = [token.lemma_ for token in comment]
    lemmatized_comment = " ".join(lemmas)

    return lemmatized_comment


def _countApologies(lemmas):
    """
    Count the number of apology lemmas in a comment.

    GIVEN:
      lemmas (str) -- lemmatized comment

    RETURN:
      num_apology_lemmas (int) -- number of apology lemmas in lemmas
    """
    num_apology_lemmas = 0
    lems = lemmas.split(" ")
    for lem in lems:
        for apology in APOLOGY_LEMMAS:
            if apology == lem:
                num_apology_lemmas += 1

    return num_apology_lemmas


def _writeNewColumns(old_file, new_file, column_1, column_2):
    """
    Write new data to disk.

    GIVEN:
      old_file (str) -- absolute path to old file
      new_file (str) -- absolute path to new file
      column_1 (list) -- list of data to be appended first (lemmatized text)
      column_2 (list) -- list of data to be appended second (count of apology lemmas)
    """
    # Insert column headers
    column_1.insert(0, "COMMENT_TEXT_LEMMATIZED")
    column_2.insert(0, "NUM_APOLOGY_LEMMAS")

    # Prepare new file for writing
    new_f = open(new_file, "w")
    csv_writer = csv.writer(new_f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
    
    # Open old file for reading
    with open(old_file, "r") as old_f:
        cnt = 0
        csv_reader = csv.reader(fixNullBytes(old_f), delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        for entry in csv_reader:
            # Append new columns to data
            new_row = entry
            new_row.extend([column_1[cnt], column_2[cnt]])
            csv_writer.writerow(new_row)
            cnt += 1

    # Close file
    new_f.close()


def _numApologyComments(apology_counts):
    cnt = 0
    for i in apology_counts[1:]: # Skip header row
        if int(i) > 0:
            cnt += 1

    return cnt


def _preprocess(old_file, new_file, src, mproc_pool):
    """
    Helper function for preprocess(). Facilitate cleaning up and lemmatizing comments, as well as
    counting apologies and writing data to disk.

    GIVEN:
      old_file (str) -- absolute path to old file
      new_file (str) -- absolute path to new file
      src (str) -- flag indicating type of data (IS, CO, PR)
      num_cpus (int) -- number of cpus to use
    """
    # Get comments
    print("    Getting comments from {}...".format(old_file))
    comments = _getRawComments(old_file)
    # Clean up comments
    print("    Cleaning up {} comment text...".format(src))
    comments = mproc_pool.map(_cleanText, comments)
    # Get comment lemmas
    print("    Lemmatizing {} comment text...".format(src))
    comment_lemmas = mproc_pool.map(_lemmatize, comments)
    # Remove stale variable
    del comments
    # Get apology counts
    print("    Counting {} apologies...".format(src))
    apology_counts = mproc_pool.map(_countApologies, comment_lemmas)
    # Write columns to disk
    print("    Writing {} to disk...".format(src))
    _writeNewColumns(old_file, new_file, comment_lemmas, apology_counts)

    #pool.close()
    #pool.join()

    # Return number of entries containing at least one apology
    return _numApologyComments(apology_counts)


def _writeApologyCounts(filename, apology_count):
    with open(filename, "w") as f:
        f.write(str(apology_count))


def preprocess(top_dir, sub_dirs, num_cpus):
    is_apology_cnt = 0
    co_apology_cnt = 0
    pr_apology_cnt = 0

    pool = mproc.Pool(num_cpus)

    for sub_dir in sub_dirs:
        curr_data_dir = os.path.join(top_dir, sub_dir)
        print("Preprocessing {}...".format(curr_data_dir))
        is_file, co_file, pr_file = getDataFilepaths(curr_data_dir)

        new_is_file = is_file.split(".csv")[0] + "_new.csv"
        new_co_file = co_file.split(".csv")[0] + "_new.csv"
        new_pr_file = pr_file.split(".csv")[0] + "_new.csv"

        aps_is_file = is_file.split(".csv")[0] + "_aps.csv"
        aps_co_file = co_file.split(".csv")[0] + "_aps.csv"
        aps_pr_file = pr_file.split(".csv")[0] + "_aps.csv"

        if os.path.exists(is_file):
            cur_is_ap_cnt = _preprocess(is_file, new_is_file, "is", pool)
            print("        # IS apologies: {}".format(cur_is_ap_cnt))
            is_apology_cnt += cur_is_ap_cnt
            _writeApologyCounts(aps_is_file, cur_is_ap_cnt)
        if os.path.exists(co_file):
            cur_co_ap_cnt = _preprocess(co_file, new_co_file, "co", pool)
            print("        # CO apologies: {}".format(cur_co_ap_cnt))
            co_apology_cnt += cur_co_ap_cnt
            _writeApologyCounts(aps_co_file, cur_co_ap_cnt)
        if os.path.exists(pr_file):
            cur_pr_ap_cnt = _preprocess(pr_file, new_pr_file, "pr", pool)
            print("        # PR apologies: {}".format(cur_pr_ap_cnt))
            pr_apology_cnt += cur_pr_ap_cnt
            _writeApologyCounts(aps_pr_file, cur_pr_ap_cnt)

        print("Cumulative IS Apologies: {}".format(is_apology_cnt))
        print("Cumulative CO Apologies: {}".format(co_apology_cnt))
        print("Cumulative PR Apologies: {}".format(pr_apology_cnt))


#### MAIN ##########################################################################################
if __name__ == "__main__":
    top_dir = canonicalize(sys.argv[1])
    num_cpus = int(sys.argv[2])
    dir_names = getSubDirNames(top_dir)
    preprocess(top_dir, dir_names, num_cpus)
