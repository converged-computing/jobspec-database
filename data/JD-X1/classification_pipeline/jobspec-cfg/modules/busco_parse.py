#!usr/bin/env python3
"""
Parse miniBUSCO output summary files and append to
the pandas dataframe containing the MAG stats.

new columns per MAG row:
    - complete_single_copy_busco (int)
    - complete_duplicated_busco (int)
    - fragmented_busco (int)
    - missing_busco (int)
    - total_busco (int)

usage:
    python3 busco_parse.py -b /path/to/miniBUSCO_outputs/MAG_ID/summary_file.txt -m /path/to/MAG_stats.csv
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
    parser = argparse.ArgumentParser(description='Parse miniBUSCO output folders and append summary stats to the pandas dataframe containing the MAG stats.')
    parser.add_argument('-b', '--busco_output', type=str, help='Path to miniBUSCO output folders.')
    parser.add_argument('-m', '--mag_stats', type=str, help='Path to MAG stats file.')
    parser.add_argument('-o', '--output', type=str, help='Path to output file.')
    return parser.parse_args()


def parse_summary(summary_file):
    """
    Parse miniBUSCO output summary file and return
    array of busco stats.

    summary_file : str
        path to miniBUSCO output summary file.
        strucutre of path:
            /path/to/miniBUSCO_outputs/MAG_ID/summary_file.txt
    """
    in_handle = open(summary_file, 'r')
    lines = in_handle.readlines()
    in_handle.close()
    out_a = np.empty(5, dtype=int)
    for line in lines:
        if line.startswith('S'):
            out_a[0] = int(line.split(", ")[1]) # single copy
        elif line.startswith('D'):
            out_a[1] = int(line.split(", ")[1]) # duplicated
        elif line.startswith('F'):
            out_a[2] = int(line.split(", ")[1]) # fragmented
        elif line.startswith('M'):
            out_a[3] = int(line.split(", ")[1]) # missing
        elif line.startswith('N'):
            out_a[4] = int(line.split(":")[1]) # total
    return out_a

def main():
    """
    Main function.

    """
    args = arg_parser()
    busco_outs =  os.listdir(args.busco_output)
    mag_stats = pd.read_csv(args.mag_stats, index_col=0)
    for i in busco_outs:
        busco_stats = parse_summary(args.busco_output + "/" + i + '/summary.txt')
        mag_stats.loc[i.strip(".fna"), 'complete_single_copy_busco'] = busco_stats[0]
        mag_stats.loc[i.strip(".fna"), 'complete_duplicated_busco'] = busco_stats[1]
        mag_stats.loc[i.strip(".fna"), 'fragmented_busco'] = busco_stats[2]
        mag_stats.loc[i.strip(".fna"), 'missing_busco'] = busco_stats[3]
        mag_stats.loc[i.strip(".fna"), 'total_busco'] = busco_stats[4]
    mag_stats.to_csv(args.output)

if __name__ == '__main__':
    main()
