#!usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import argparse
"""
# parse per-mag xgboost output files
# append to mag stats file
# 
# xgb_output files art tab delimited
# basic structure:
#    MAG_subset_ID    phage, plasmid, prokaryote, microeukaryotes
#
#
# new columns in mag_stats per MAG row:
#   - xgb_phage (float)
#   - xgb_plasmid (float)
#   - xgb_prokaryote (float)
#   - xgb_microeukaryotes (float)

# usage:
#    python3 xg_parse.py -x /path/to/xgb_outputs -m /path/to/MAG_stats.csv -o /path/to/output.csv

"""


def arg_parser():
    """
    Parse arguments from command line.

    Returns
    -------
    argparse.ArgumentParser
        Object containing arguments passed at command line.
    """
    parser = argparse.ArgumentParser(description='Parse eukcc output files and append to mag stats file.')
    parser.add_argument('-x', '--xg_output', type=str, help='Path to eukcc output files.')
    parser.add_argument('-m', '--mag_stats', type=str, help='Path to MAG stats file.')
    parser.add_argument('-o', '--output', type=str, help='Path to output file.')
    return parser.parse_args()



def main():
    """
    Main function for xg_parse.py
    """
    args = arg_parser()
    mag_stats = pd.read_csv(args.mag_stats, index_col=0)
    xg_output = args.xg_output
    mags = os.listdir(xg_output)
    for mag in mags:
        xg_file = pd.read_csv(os.path.join(xg_output, mag), sep='\t', header=None)
        mag_stats.loc[mag.strip(".fna.out"), "xgb_phage"] = xg_file[1].mean()
        mag_stats.loc[mag.strip(".fna.out"), "xgb_plasmid"] = xg_file[2].mean()
        mag_stats.loc[mag.strip(".fna.out"), "xgb_prokaryote"] = xg_file[3].mean()
        mag_stats.loc[mag.strip(".fna.out"), "xgb_microeukaryotes"] = xg_file[4].mean()
    mag_stats.to_csv(args.output)

if __name__ == '__main__':
    main()

