#!usr/bin/env python3

"""
Parse eukcc output files and append to mag stats file

new columns per MAG row:
    - suggested taxonomy (str)
    - completeness (float)
    - contamination (float)

usage:
    python3 eukcc_parse.py -e /path/to/eukcc_outputs -m /path/to/MAG_stats.csv -o /path/to/output.csv
"""

import os
import numpy as np
import pandas as pd
import argparse


def arg_parser():
    """
    Parse arguments from command line.

    Returns
    -------
    argparse.ArgumentParser
        Object containing arguments passed at command line.
    """
    parser = argparse.ArgumentParser(description='Parse eukcc output files and append to mag stats file.')
    parser.add_argument('-e', '--eukcc_output', type=str, help='Path to eukcc output files.')
    parser.add_argument('-m', '--mag_stats', type=str, help='Path to MAG stats file.')
    parser.add_argument('-o', '--output', type=str, help='Path to output file.')
    return parser.parse_args()

def get_clade(log_file):
    """
    Parse eukcc log file and return clade info

    log_file : str
        path to eukcc log file
        structure of path:
            /path/to/eukcc_outputs/MAG_ID/eukcc.log
    """
    for line in log_file:
        if "Genome belongs to clade" in line:
            clade = line.split(": ")[2]
            clade = clade.split(" (")[0]
    return clade


def main():
    args = arg_parser()
    mag_stats = pd.read_csv(args.mag_stats, index_col=0)
    eukcc_output = args.eukcc_output
    mags = os.listdir(eukcc_output)
    for mag in mags:
        log_file = open(eukcc_output + "/" + mag + "/eukcc.log", 'r').readlines()
        clade = get_clade(log_file)
        print(clade)
        # extract second row of summary file an np array
        completeness_array = pd.read_csv(str(eukcc_output + "/" + mag + "/eukcc.csv"), sep = "\t").to_numpy()
        # append the clade and the completeness array that matches the first column in the mag stats csv
        completeness_array[0][0] = completeness_array[0][0].split("/")[-1]
        mag_stats.loc[mag.strip(".fna"), 'EukCC_clade'] = clade
        mag_stats.loc[mag.strip(".fna"), 'EukCC_completeness'] = completeness_array[0][1]
        mag_stats.loc[mag.strip(".fna"), 'EukCC_contamination'] = completeness_array[0][2]
    mag_stats.to_csv(args.output)

if __name__ == "__main__":
    main()