#!/usr/bin/env python3


import argparse
import shutil
import os
import sys

from rse.utils.command import Command
import rse.utils.file as utils

import fnmatch
import tempfile
import requests

# Note that we are doing with the web UI because the programmatic rest API
# limits queries per day
here = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(here)


def get_parser():
    parser = argparse.ArgumentParser(description="Job Specification Downloader")
    parser.add_argument(
        "--input",
        default=os.path.join(root, "data"),
        help="input directory with repository names",
    )
    return parser


def recursive_find(base, pattern="*"):
    for root, _, filenames in os.walk(base):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def clone(url, dest):
    dest = os.path.join(dest, os.path.basename(url))
    cmd = Command("git clone --depth 1 %s %s" % (url, dest))
    cmd.execute()
    if cmd.returncode != 0:
        print("Issue cloning %s" % url)
        return
    return dest


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if not args.input:
        sys.exit("An input directory is required.")

    # Create a base temporary folder to work from
    tempdir = tempfile.mkdtemp()

    # Save associations
    config_files = {}
    config_file_json = os.path.join(here, "data", "jobspec-config-files.json")
    if os.path.exists(config_file_json):
        config_files = utils.read_json(config_file_json)

    # Do a total count first
    total = 0
    for org in os.listdir(args.input):
        repo_dir = os.path.join(args.input, org)
        for repo in os.listdir(repo_dir):
            total += 1

    # Read in text, and as we go, generate content hash.
    # We don't want to use duplicates (forks are disabled, but just being careful)
    count = 0
    for org in os.listdir(args.input):
        repo_dir = os.path.join(args.input, org)
        for repo in os.listdir(repo_dir):
            uri = f"{org}/{repo}"
            print(f"👀️ Inspecting {uri}: {count} of {total}")
            count += 1

            scripts = list(recursive_find(repo_dir, "*.*"))
            config_dir = os.path.join(repo_dir, repo, "jobspec-cfg")
            if uri in config_files:
                continue
            config_files[uri] = {}

            # Do a quick check for 404 - if not OK, we won't check
            url = f"https://github.com/{org}/{repo}"
            response = requests.head(url)
            if response.status_code == 404:
                print("    Not found (or no longer public)")
                continue

            dest = None
            try:
                # Try clone (and cut out early if not successful)
                dest = clone(url, tempdir)
                if not dest:
                    continue
            except:
                print(f"Issue with {url}, skipping")
                continue

            for script in scripts:
                relative_path = os.path.relpath(script, repo_dir)
                repo_path = os.path.join(dest, script)
                # Bad file codec usually
                try:
                    content = "\n".join(utils.read_file(repo_path))
                except:
                    continue
                dir_path = os.path.dirname(repo_path)

                # Tokenize the script, look for paths in the repository
                tokens = [
                    x.strip()
                    for x in content.replace("\n", " ").split(" ")
                    if x.strip()
                ]
                for token in tokens:
                    token = token.strip(os.sep)

                    # Must be under dest clone
                    search_path = os.path.join(dest, token)
                    if os.path.exists(search_path) and not os.path.isdir(search_path):
                        print(
                            f"  => Found associated file for {relative_path}: {token}"
                        )
                        if relative_path not in config_files[uri]:
                            config_files[uri][relative_path] = []
                        if token not in config_files[uri][relative_path]:
                            config_files[uri][relative_path].append(token)

                        # Give a name we can reliably parse
                        dest_path = os.path.join(config_dir, token)
                        dest_path_dir = os.path.dirname(dest_path)
                        if not os.path.exists(dest_path_dir):
                            os.makedirs(dest_path_dir)
                        shutil.copyfile(search_path, dest_path)

            if os.path.exists(dest):
                shutil.rmtree(dest)

            print()
            utils.write_json(config_files, config_file_json)

    # Do a final cleaning and file found
    total = 0
    scripts = 0
    keepers = {}
    possible_lammps = []
    for uri, repo in config_files.items():
        if not repo:
            continue
        scripts += len(repo)
        keepers[uri] = repo
        for _, items in repo.items():
            total += len(items)
            for item in items:
                if item.endswith(".in"):
                    possible_lammps.append(item)

    print(len(keepers))
    print(len(config_files))
    print(total)

    # len(keepers) is 3309
    # len(config_files) is 11535
    # This means ~28.7% of repos had at least one associated file
    utils.write_json(keepers, config_file_json)


if __name__ == "__main__":
    main()
