import argparse
import numpy as np
from extractors import *
import logging
from os import access
from os import R_OK
from os import W_OK
from os.path import isfile
from os.path import isdir
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--extractors", nargs="+", required=True,
                    type=str, help="Constructor definitions of extractors.")
parser.add_argument("-i", "--image_list", required=True, type=str,
                    help="Path to a file with paths to images to be extracted.")
parser.add_argument("-o", "--output_dir", required=True,
                    type=str, help="Path to an output folder.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Verbosity level [on/off].", default=False)
parser.add_argument("-ev", "--extra_verbose", action="store_true",
                    help="Extra verbosity level [on/off].", default=False)
parser.add_argument("--batch_size", type=int,
                    help="""The size of the batch in image processing. 
    If the extraction is too slow, you should increase the batch size.
    If the extarction is raising memory error, you should decrease the batch size.
    Default value is 64.
    """,
                    default=64)


def validate_args(args):
    passed = True
    if len(args.extractors) == 0:
        logging.error("Missing extractors (--extractors).")
        passed = False

    for ext in args.extractors:
        try:
            extractor = eval(ext)
            del extractor
        except BaseException as err:
            logging.error(f"Invalid extractor name: {ext}.")
            logging.error(f"ERR: {err}")
            passed = False

    if not isfile(args.image_list) or not access(args.image_list, R_OK):
        logging.error(
            f"The file {args.image_list} does not exist or is not readable.")
        passed = False

    if not isdir(args.output_dir) or not access(args.output_dir, W_OK):
        logging.error(
            f"The output directory {args.output_dir} does not or is not writable.")
        passed = False

    if args.batch_size < 1:
        logging.error(
            f"The batch size must be atleast 1. Current value = {args.batch_size}")
        passed = False

    return passed


def process_extraction(args):
    images_paths = []
    with open(args.image_list, "r") as file:
        images_paths = [line.rstrip() for line in file.readlines()]

    logging.debug("Image list read. Path sample: ")
    logging.debug(images_paths[:3])

    for ext_name in args.extractors:
        extractor = eval(ext_name)
        extractor = BatchExtractor(
            args.batch_size, extractor, show_progress=True)
        logging.warning(f"Extracting with {ext_name}")
        features = extractor(images_paths)
        file_name = ext_name.replace("(", "_").replace(")", "_").replace("\"", "").replace("=", ":").replace("/", "-") + ".npy"
        with open(os.path.join(args.output_dir, file_name), 'wb') as f:
            np.save(f, features)
        logging.warning(f"Extracting with {ext_name} DONE\n")
    logging.warning("Extracting DONE")


def main(args):
    if args.extra_verbose:
        logging.basicConfig(
            format='%(levelname)s:\t%(message)s', level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(
            format='%(levelname)s:\t%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(
            format='%(levelname)s:\t%(message)s', level=logging.WARNING)

    logging.info("\n")
    logging.info("*** Validating arguments ***")
    logging.debug(args)
    valid = validate_args(args)
    if not valid:
        logging.error("Could not continue due to the errors. Stopping!")
        sys.exit(0)

    logging.info("Arguments OK")
    logging.info("\n")
    logging.info("*** Starting extraction ***")
    start = time.time()
    process_extraction(args)
    duration = time.time() - start
    logging.info(f"DONE in {duration}s")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
