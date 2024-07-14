#!/usr/bin/env python3

#### PYTHON IMPORTS ################################################################################
import csv
import os
import shutil
import sys
from pathlib import Path


#### CONFIG ########################################################################################
csv.field_size_limit(sys.maxsize)


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
    """
    return os.path.realpath(top_dir)


def _overwrite(old_file, new_file): # pragma: no cover
    """
    Helper function for deduplicate(). Overwrite old_file with new_file.

    GIVEN:
      old_file (str) -- path to file to overwrite
      new_file (str) -- path to file to overwrite old_file with

    RETURN:
      None
    """
    # Delete old_file
    if os.path.exists(old_file):
        os.remove(old_file)

    # Rename new_file to old_file
    if os.path.exists(new_file):
        os.rename(new_file, old_file)


def _deduplicate(old_file, new_file, header):
    """
    Helper function for deduplicate(). Remove duplicate header rows from downloaded CSV files.

    GIVEN:
      old_file (str) -- path to downloaded file
      new_file (str) -- path to store deduplicated old_file
      header (list) -- header row to deduplicate

    RETURN:
      None
    """
    if os.path.exists(old_file):
        # Set up dedup file access
        Path(new_file).touch()
        f_dedup = open(new_file, "a")
        dedup_writer = csv.writer(f_dedup, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
        # Keep track of whether we've seen the header row before
        seen_header = False
        # Open file with duplicates
        with open(old_file, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(fixNullBytes(f), delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
            for entry in csv_reader:
                # If the current entry is the header row
                if entry == header:
                    print("\t\tFound header!")
                    # If we haven't seen the header row before, write it
                    if not seen_header:
                        print("\t\tWriting header!")
                        dedup_writer.writerow(entry)
                        seen_header = True
                    else:
                        print("\t\tSkipping header!")
                        pass
                else:
                    dedup_writer.writerow(entry)
        f_dedup.close()
    else:
        pass


def deduplicate(data_dir, sub_dirs, overwrite=False):
    """
    Remove duplicate header rows from downloaded CSV files.

    GIVEN:
      data_dir (str) -- path to downloaded data
      sub_dirs (list) -- list of directory names
      overwrite (bool) -- whether or not to overwrite the old data files

    RETURN:
      None
    """
    for sub_dir in sub_dirs:
        temp_data_dir = os.path.join(data_dir, sub_dir)
        print("\tDeduplicating {}...".format(temp_data_dir))
        issues_file, commits_file, pull_requests_file = getDataFilepaths(temp_data_dir)

        dedup_issues_file = issues_file.split(".csv")[0] + "_dedup.csv"
        dedup_commits_file = commits_file.split(".csv")[0] + "_dedup.csv"
        dedup_pull_requests_file = pull_requests_file.split(".csv")[0] + "_dedup.csv"

        _deduplicate(issues_file, dedup_issues_file, ISSUES_HEADER)
        _deduplicate(commits_file, dedup_commits_file, COMMITS_HEADER)
        _deduplicate(pull_requests_file, dedup_pull_requests_file, PULL_REQUESTS_HEADER)

        if overwrite:
            _overwrite(issues_file, dedup_issues_file)
            _overwrite(commits_file, dedup_commits_file)
            _overwrite(pull_requests_file, dedup_pull_requests_file)


def getSubDirNames(top_dir):
    """
    Get the names of subdirectories, one level deep.
    """
    return [d[1] for d in os.walk(top_dir)][0]


def moveFiles(top_dir, dir_names):
    """
    Rename files and move them to the top_dir.
    """
    for sub_dir in dir_names:
        co_file = os.path.join(top_dir, sub_dir, "commits/commits.csv")
        is_file = os.path.join(top_dir, sub_dir, "issues/issues.csv")
        pr_file = os.path.join(top_dir, sub_dir, "pull_requests/pull_requests.csv")

        if os.path.exists(co_file):
            new_co_file = os.path.join(top_dir, "co_{}.csv".format(sub_dir))
            os.rename(co_file, new_co_file)
            print("\tMoved {} --> {}".format(co_file, new_co_file))

        if os.path.exists(is_file):
            new_is_file = os.path.join(top_dir, "is_{}.csv".format(sub_dir))
            os.rename(is_file, new_is_file)
            print("\tMoved {} --> {}".format(is_file, new_is_file))

        if os.path.exists(pr_file):
            new_pr_file = os.path.join(top_dir, "pr_{}.csv".format(sub_dir))
            os.rename(pr_file, new_pr_file)
            print("\tMoved {} --> {}".format(pr_file, new_pr_file))


def deleteEmptyDirs(top_dir, dir_names):
    """
    Delete empty directories.
    """
    for sub_dir in dir_names:
        temp_path = os.path.join(top_dir, sub_dir)
        shutil.rmtree(temp_path)
        print("\tDeleted {}".format(temp_path))


#### MAIN ##########################################################################################
if __name__ == "__main__":
    top_dir = canonicalize(sys.argv[1])
    dir_names = getSubDirNames(top_dir)
    print("Deduplicating headers...")
    deduplicate(top_dir, dir_names, overwrite=True)
    print("Moving files...")
    moveFiles(top_dir, dir_names)
    print("Deleting directories...")
    deleteEmptyDirs(top_dir, dir_names)
