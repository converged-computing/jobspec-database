#!/usr/bin/env python3

import argparse
import os
import random
import sys
from datetime import datetime
from time import sleep

import rse.utils.file as utils
from bs4 import BeautifulSoup
from rse.utils.command import Command
from selenium import webdriver

# Note that we are doing with the web UI because the programmatic rest API
# limits queries per day
here = os.path.abspath(os.path.dirname(__file__))

# The drivers must be on path
drivers = os.path.join(here, "drivers")
os.environ["PATH"] = "%s:%s" % (drivers, os.environ["PATH"])
sys.path.insert(0, drivers)

token = os.environ.get("GITHUB_TOKEN")
if not token:
    sys.exit("Please export GITHUB_TOKEN")


def clone(url, dest):
    dest = os.path.join(dest, os.path.basename(url))
    cmd = Command("git clone --depth 1 %s %s" % (url, dest))
    cmd.execute()
    if cmd.returncode != 0:
        print("Issue cloning %s" % url)
        return
    return dest


def confirm_action(question):
    """
    Ask for confirmation of an action
    """
    response = input(question + " (yes/no)? ")
    while len(response) < 1 or response[0].lower().strip() not in "ynyesno":
        response = input("Please answer yes or no: ")
    if response[0].lower().strip() in "no":
        return False
    return True


def get_parser():
    parser = argparse.ArgumentParser(description="Dockerfile Scraper")
    parser.add_argument(
        "--start-date",
        default="2013-04-11",
        help="starting date",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(here, "data"),
        help="output data directory for results",
    )
    parser.add_argument(
        "--days",
        default=100,
        help="days to search for repos over",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Output file name is the date
    date = datetime.now().strftime("%Y-%m-%d")
    outfile = f"raw-links-{date}.json"
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    save_to = os.path.join(args.outdir, outfile)

    # Save all slurm job links
    links = {}
    if os.path.exists(save_to):
        links = utils.read_json(save_to)

    # Create the driver
    driver = webdriver.Chrome()

    # This will connect to currently open browser
    # We will want to parse pages from this URL
    confirm_action("Browse to GitHub.com and login, and another 'yes'")

    # Allow timeout
    driver.set_page_load_timeout(10)

    # These are different job submission directives
    directives = [
        "SBATCH",
        "PBATCH",
        "PBS",
        "OAR",
        "FLUX",
    ]

    # Let's do a matrix across analyses and then HPC terms
    # This might give duplicates, which is OK
    hpc_terms = [
        "gpu",
        "array",
        "module",
        "mpi",
        "cuda",
        "intel",
    ]

    # We can only get 100 results per set of terms. Boo
    app_terms = [
        "lammps",
        "python",
        "amd",
        "amg",
        "rust",
        "spack",
        "easybuild",
        "resnet",
        "pytorch",
        "nvidia",
        "tensorflow",
        "pmix",
        "kripke",
        "quicksilver",
        "maestro",
        "snakemake",
        "nextflow",
        "dask",
        "kube",
        "docker",
        "singularity",
        "julia",
        "matlab",
        "nccl",
        "ray",
        "fio",
        "ior",
    ]

    # Create matrix of combined terms
    terms = []
    for app_term in app_terms:
        for hpc_term in hpc_terms:
            terms.append(f"{app_term}+{hpc_term}")

    for directive in directives:
        directive_url = f"https://github.com/search?q=%22%23{directive}%22"
        for term in terms:
            uid = f"{directive}-{term}"
            if uid in links:
                continue

            print(f"⭐️ NEW TERM: {term}")
            base_url = f"{directive_url}+{term}++language%3AShell&type=code"
            links[uid] = []

            page = 1
            while True:
                print(f"Parsing page {page}")
                url = f"{base_url}&p={page}"
                driver.get(url)

                # Sleep at least one second
                sleep(random.choice(range(0, 1000)) * 0.001 + 1)

                # This gets the whole page - this is easier to parse than with selenium
                body = driver.execute_script(
                    "return document.documentElement.outerHTML;"
                )

                # Use beautiful soup to parse it
                soup = BeautifulSoup(body, "html.parser")

                # Find all the direct links to save
                new_links = soup.findAll(
                    "a", attrs={"data-testid": "link-to-search-result"}
                )

                if not new_links:
                    break

                for link in new_links:
                    links[uid].append(link.get("href"))

                # Save the results as we go
                utils.write_json(links, save_to)
                page += 1
                print(f"Found {len(links)} total job file links (saving incrementally)")

    # One more save...
    utils.write_json(links, save_to)

    print("CHECK RESULTS")
    import IPython

    IPython.embed()
    sys.exit()


if __name__ == "__main__":
    main()
