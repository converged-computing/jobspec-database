#!/usr/bin/env python3


#### PYTHON IMPORTS ################################################################################
import argparse
import multiprocessing as mproc
import sys


#### PROJECT IMPORTS ###############################################################################
from src.apologies import classify
from src.deduplicate import deduplicate
from src.delete import delete
from src.developers import developerStats
from src.download import download
from src.graphql import getRateLimitInfo
from src.helpers import canonicalize, doesPathExist, GITHUB_LANGUAGES
from src.info import infoData
from src.preprocess import preprocess
from src.random import randomSample
from src.search import search, topRepos
from src.stats import stats


#### GLOBALS #######################################################################################
ASSERT_NOT_EXIST = "The provided {}={} does not exist."


#### FUNCTIONS #####################################################################################
def downloadCommand(args):
    """
    Parse arguments for 'download' command and pass them to src.download:download().
    """
    # Canonicalize filepaths
    args.repo_file = canonicalize(args.repo_file)
    args.data_dir = canonicalize(args.data_dir)

    # Check assertions
    assert doesPathExist(args.repo_file), ASSERT_NOT_EXIST.format("repo_file", args.repo_file)
    assert doesPathExist(args.data_dir), ASSERT_NOT_EXIST.format("data_dir", args.data_dir)

    # Pass arguments to src.download:download()
    download(args.repo_file, args.data_dir, args.data_types)


def dedupCommand(args):
    """
    Parse arguments for `dedup` command and pass them to src.deduplicate:deduplicate().
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)

    # Check assertions
    assert doesPathExist(args.data_dir), ASSERT_NOT_EXIST.format("data_dir", args.data_dir)

    # Pass arguments to src.deduplicate:deduplicate()
    deduplicate(args.data_dir, args.overwrite)


def deleteCommand(args):
    """
    Delete the CSV data stored in the given data_dir.
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)

    # Verify with user
    input("Are you sure you want to delete the data in {}? Press CTRL+C now to abort, or any key to "
          "continue with deletion.".format(args.data_dir))

    # Pass arguments to src.delete:delete()
    delete(args.data_dir)


def developerStatsCommand(args):
    """
    Compute stats for developers.
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)

    if args.num_procs == 0:
        args.num_procs = mproc.cpu_count()

    # Pass arguments to src.developers:developerStats()
    developerStats(args.data_dir, args.num_procs)


def searchCommand(args):
    """
    Parse arguments for 'search' command and pass them to src.search:search().
    """
    # Check assertions
    assert args.stars >= 0, "Argument 'stars' must be greater than or equal to 0."
    assert args.total >= 0, "Argument 'total' must be greater than or equal to 0."
    assert False if args.term == "" and args.stars == 0 and args.language == "None" else True, \
        "Argument 'term' cannot be empty when argument 'stars'=0 and argument 'languages'='None'."
    assert False if args.save == True and args.results_file is None else True, \
        "Argument '--results_file' must be provided when argument `--save` is provided."

    # Canonicalize filepaths
    if args.results_file is not None:
        args.results_file = canonicalize(args.results_file)

    # Fix total
    if args.total > 1000:
        args.total = 1000

    # Pass arguments to src.search:search()
    search(args.term, args.stars, args.language, args.total, args.save, args.results_file)


def topReposCommand(args):
    """
    Parse arguments for 'top_repos' command and pass them to src.search:topRepos().
    """
    # Check assertions
    assert args.stars >= 0, "Argument 'stars' must be greater than or equal to 0."
    
    # Canonicalize filepaths
    args.results_file = canonicalize(args.results_file)

    # Pass arguments to src.search:topRepos().
    topRepos(args.languages, args.stars, args.results_file)


def preprocessCommand(args):
    """
    Parse arguments for 'preprocess' command and pass them to src.preprocess:preprocess().
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)

    # Check assertions
    assert doesPathExist(args.data_dir), ASSERT_NOT_EXIST.format("data_dir", args.data_dir)
    assert args.num_procs <= mproc.cpu_count(), \
        "Argument 'num_procs' cannot be greater thanmaximum number of CPUs: {}.".format(mproc.cpu_count())

    if args.num_procs == 0:
        args.num_procs = mproc.cpu_count()

    # Pass arguments to src.preprocess:preprocess().
    preprocess(args.data_dir, args.num_procs, args.overwrite)


def classifyCommand(args):
    """
    Parse arguments for 'classify' command and pass them to src.apologies:classify().
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)

    # Check assertions
    assert doesPathExist(args.data_dir), ASSERT_NOT_EXIST.format("data_dir", args.data_dir)
    assert args.num_procs <= mproc.cpu_count(), \
        "Argument 'num_procs' cannot be greater than maximum number of CPUs: {}.".format(mproc.cpu_count())

    if args.num_procs == 0:
        args.num_procs = mproc.cpu_count()

    # Pass arguments to src.apologies:classify().
    classify(args.data_dir, args.num_procs, args.overwrite)


def infoDataCommand(args):
    """
    Display useful information about the data in a directory.
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)

    # Check assertions
    assert doesPathExist(args.data_dir), ASSERT_NOT_EXIST.format("data_dir", args.data_dir)
    
    # Pass arguments to src.info:infoData()
    infoData(args.data_dir)


def infoRateLimitCommand(args):
    """
    Query GitHub's GraphQL API for rate limit information.
    """
    print(getRateLimitInfo())


def randomSampleCommand(args):
    """
    Select a random sample of developer comments.
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)
    args.output_file = canonicalize(args.output_file)

    # Check assertions
    assert doesPathExist(args.data_dir), ASSERT_NOT_EXIST.format("data_dir", args.data_dir)
    assert args.size > 0, "Argument 'size'={} cannot be less than 1.".format(args.size)

    # Verify overwriting
    if doesPathExist(args.output_file):
        input("The output_file='{}' already exists. Do you wish to overwrite it? Press CTRL+C now "
              "to abort, or any key to continue and overwrite.".format(args.output_file))

    # Pass arguments to src.random:randomSample()
    randomSample(args.data_dir, args.size, args.apologies_only, args.source, args.output_file,
        args.export_all)


def statsCommand(args):
    """
    Compute statistics for apologies and non-apologies.
    """
    # Canonicalize filepaths
    args.data_dir = canonicalize(args.data_dir)

    # Check assertions
    assert doesPathExist(args.data_dir), ASSERT_NOT_EXIST.format("data_dir", args.data_dir)
    assert args.num_procs <= mproc.cpu_count(), \
        "Argument 'num_procs' cannot be greater than maximum number of CPUs: {}.".format(mproc.cpu_count())

    if args.num_procs == 0:
        args.num_procs = mproc.cpu_count()

    # Pass arguments to src.stats:stats()
    stats(args.data_dir, args.num_procs)


#### MAIN ##########################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scripts to facilitate downloading data from GitHub, loading it into an HDF5 "
        "file, and analyzing the data."
    )

    command_parsers = parser.add_subparsers(
        help="Available commands."
    )

    #### DOWNLOAD COMMAND
    download_parser = command_parsers.add_parser(
        "download", help="Download data from GitHub repositories using the GraphQL API."
    )

    download_parser.add_argument(
        "repo_file", type=str, help="The path for a file containing GitHub repository URLs to "
        "download data from. This file should contain a single repo URL per line. Relative paths "
        "will be canonicalized."
    )
    download_parser.add_argument(
        "data_types", type=str, choices=["issues", "pull_requests", "commits", "all"],
        help="The type of data to download from the given repositories. All of the relevant "
        "metadata, including comments will be downloaded for whichever option is given."
    )
    download_parser.add_argument(
        "data_dir", type=str, help="The path for a directory to save the downloaded data to. "
        "Relative paths will be canonicalized. Downloaded data will be in the form of a CSV file "
        "and placed in a subdirectory, e.g. data_dir/issues/."
    )
    download_parser.set_defaults(func=downloadCommand)

    #### DEDUPLICATE COMMAND
    dedup_parser = command_parsers.add_parser(
        "dedup", help="Remove duplicate header rows from downloaded data."
    )

    dedup_parser.add_argument(
        "data_dir", type=str, help="The path to a directory where data is downloaded and ready to "
        "be deduplicated. Relative paths will be canonicalized."
    )
    dedup_parser.add_argument(
        "--overwrite", default=False, action="store_true", help="If included, the deduplicated "
        "CSV file will overwrite the old CSV file. Using this flag is not recommended unless you "
        "have backups of your data."
    )
    dedup_parser.set_defaults(func=dedupCommand)

    #### DELETE COMMAND
    delete_parser = command_parsers.add_parser(
        "delete", help="Delete local CSV data from disk. This command cannot be used to delete "
        "the HDF5 file."
    )
    
    delete_parser.add_argument(
        "data_dir", type=str, help="The path for a directory containing downloaded data. Relative "
        "paths will be canonicalized."
    )
    delete_parser.set_defaults(func=deleteCommand)

    #### SEARCH COMMAND
    search_parser = command_parsers.add_parser(
        "search", help="Search GitHub for a list of repositories based on provided criteria."
    )

    search_parser.add_argument(
        "term", type=str, help="Filter results to only include those that match this search term. "
        "Enter '' to remove this filter. Note that this cannot be empty when 'stars'=0 and "
        "'languages'='None' due to a limitation in the API."
    )
    search_parser.add_argument(
        "stars", type=int, help="Filter out repositories with less than this number of stars. Enter"
        " '0' to remove this filter."
    )
    search_parser.add_argument(
        "total", type=int, help="Return this many repositories (or less if filters are restrictive."
        "Enter '0' to remove this filter. Note that the maximum number of results that can be "
        "returned is 1000 due to a limitation in the API."
    )
    search_parser.add_argument(
        "language", type=str, choices=GITHUB_LANGUAGES, help="Filter results to only include "
        "repositories using this language. Enter 'None' to remove this filter."
    )
    search_parser.add_argument(
        "--save", default=False, action="store_true", help="Whether or not to save the list of "
        "repositories to disk."
    )
    search_parser.add_argument(
        "--results_file", type=str, help="The name of the file to save results to. Relative paths "
        "will be canonicalized. This option is ignored when --save=False."
    )
    search_parser.set_defaults(func=searchCommand)

    #### TOP_REPOS COMMAND
    top_repos_parser = command_parsers.add_parser(
        "top_repos", help="Download the top 1000 repo URLs for each of the languages specified."
    )

    top_repos_parser.add_argument(
        "languages", type=str, choices=["tiobe_index", "github_popular", "combined"],
        help="The list of languages to get repo URLs for. Either the top 50 from the TIOBE Index, "
        "the most popular from GitHub, or the combined set of languages from both."
    )
    top_repos_parser.add_argument(
        "stars", type=int, help="Filter out repositories with less than this number of stars. Enter"
        " '0' to remove this filter."
    )
    top_repos_parser.add_argument(
        "results_file", type=str, help="The name of the file to save URLs to. Relative paths will "
        "be canonicalized."
    )
    top_repos_parser.set_defaults(func=topReposCommand)

    #### PREPROCESS COMMAND
    preprocess_parser = command_parsers.add_parser(
        "preprocess", help="For each downloaded CSV file, append a 'COMMENT_TEXT_LEMMATIZED' column"
        " that contains text that (1) is lowercased, (2) has punctuation removed, (3) has non-space"
        " whitespace removed, and (4) is lemmatized."
    )

    preprocess_parser.add_argument(
        "data_dir", type=str, help="The path to a directory where data is downloaded and ready to "
        "be preprocessed. Relative paths will be canonicalized."
    )
    preprocess_parser.add_argument(
        "num_procs", type=int, help="Number of processes (CPUs) to use for multiprocessing. Enter "
        "'0' to use all available CPUs."
    )
    preprocess_parser.add_argument(
        "--overwrite", default=False, action="store_true", help="If included, the lemmatized "
        "CSV file will overwrite the old CSV file. Using this flag is not recommended unless you "
        "have backups of your data."
    )
    preprocess_parser.set_defaults(func=preprocessCommand)

    #### CLASSIFY COMMAND
    classify_parser = command_parsers.add_parser(
        "classify", help="For each dataset in the given HDF5 file, append a 'NUM_POLOGY_LEMMAS' "
        " column that contains the total number of apology lemmas in the 'COMMENT_TEXT_LEMMATIZED' "
        " column."
    )
    
    classify_parser.add_argument(
        "data_dir", type=str, help="The path to a directory with data that has already been "
        "populated using the 'downloadload' and 'preprocess' commands. Relative paths will be"
        " canonicalized."
    )
    classify_parser.add_argument(
        "num_procs", type=int, help="Number of processes (CPUs) to use for multiprocessing. Enter "
        "'0' to use all available CPUs."
    )
    classify_parser.add_argument(
        "--overwrite", default=False, action="store_true", help="If included, the classified "
        "CSV file will overwrite the old CSV file. Using this flag is not recommended unless you "
        "have backups of your data."
    )
    classify_parser.set_defaults(func=classifyCommand)

    #### INFO_DATA COMMAND
    info_data_parser = command_parsers.add_parser(
        "info_data", help="Display info about the downloaded data."
    )

    info_data_parser.add_argument(
        "data_dir", type=str, help="The path for a directory containing downloaded data. Relative "
        "paths will be canonicalized."
    )
    info_data_parser.set_defaults(func=infoDataCommand)

    #### INFO_RATE_LIMIT COMMAND
    info_rate_limit_parser = command_parsers.add_parser(
        "info_rate_limit", help="Display rate limiting info from GitHub's GraphQL API."
    )
    info_rate_limit_parser.set_defaults(func=infoRateLimitCommand)

    #### RANDOM_SAMPLE COMMAND
    random_parser = command_parsers.add_parser(
        "random_sample", help="Select a random sample of developer comments based on the provided "
        "criteria."
    )

    random_parser.add_argument(
        "data_dir", type=str, help="The path for a directory containing processed and classified "
        "data. Relative paths will be canonicalized."
    )
    random_parser.add_argument(
        "size", type=int, help="The desired random sample size."
    )
    random_parser.add_argument(
        "source", type=str, choices=["IS", "CO", "PR", "ALL"], help="The source of the data to "
        "sample from; IS=issues, CO=commits, PR=pull requests, ALL=all saources."
    )
    random_parser.add_argument(
        "output_file", type=str, help="The path for a file to save the random sample to, if "
        "--export_all=False, otherwise the directory to save random samples to."
    )
    random_parser.add_argument(
        "--apologies_only", default=False, action="store_true", help="If included, random samples "
        "will only be collected from comments classified as apologies."
    )
    random_parser.add_argument(
        "--export_all", default=False, action="store_true", help="Whether or not to export as many"
        " samples of the given size or just one sample."
    )
    random_parser.set_defaults(func=randomSampleCommand)

    #### APOLOGY_STATS COMMAND
    stats_parser = command_parsers.add_parser(
        "apology_stats", help="Compute statistics for: (1) average apology length, (2) average non"
        "-apology length, (3) average lemmas per apology, (4) frequency of lemmas."
    )

    stats_parser.add_argument(
        "data_dir", type=str, help="The path for a directory containing processed and classified "
        "data. Relative paths will be cononicalized."
    )
    stats_parser.add_argument(
        "num_procs", type=int, help="Number of processes (CPUs) to use for multiprocessing. Enter "
        "'0' to use all available CPUs."
    )
    stats_parser.set_defaults(func=statsCommand)

    #### DEVELOPER_STATS COMMAND
    developer_parser = command_parsers.add_parser(
        "developer_stats", help="Compute statistics for developers."
    )

    developer_parser.add_argument(
        "data_dir", type=str, help="The path for a directory containing processed and classified "
        "data. Relative paths will be canonicalized."
    )
    developer_parser.add_argument(
        "num_procs", type=int, help="Number of processes (CPUs) to use for multiprocessing. Enter "
        "'0' to use all available CPUs."
    )
    developer_parser.set_defaults(func=developerStatsCommand)

    #### PARSER ARGUMENTS
    args = parser.parse_args()
    print(args)
    args.func(args)


