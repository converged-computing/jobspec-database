#!usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
from Bio import SeqIO


"""
Usage:
    python3 mag.stats.py -f /path/to/fasta/files
"""

def arg_parser():
    """
    Parse arguments from command line.
    
    Returns
    -------
    argparse.ArgumentParser
        Object containing arguments passed at command line.
    """
    parser = argparse.ArgumentParser(description='Calculate N and GC percentages and sequence lengths for a fasta file.')
    parser.add_argument('-f', '--fasta_repo', type=str, help='Path to directory with fasta files.')
    return parser.parse_args()

def parser(fasta_file):
    """
    Parse one fasta file with Biopython SeqIO.
    
    Parameters
    ----------  
    fasta_file : str
        Path to fasta file.
    
    Returns
    -------
    dict
        Dictionary of sequences.
    """
    seqs = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        seqs[record.id] = str(record.seq)
    return seqs
    

def get_n_perc(fasta_dict):
    """
    Calculate N percentage of sequences in a SeqIO
    parsed and fasta file recast as a dictionary
    object. Returns numpy array of n content.
    
    Parameters
    ----------
    fasta_dict: dict
        Dictionary of sequences.

    Returns
    -------
    numpy.ndarray  
        Array of N percentages.
    """
    n_perc = []
    for key in fasta_dict:
        n_perc.append(fasta_dict[key].count('N') / len(fasta_dict[key]))
    return np.array(n_perc)

def get_gc_perc(fasta_dict):
    """
    Calculate GC percentage of sequences in a SeqIO
    parsed and fasta file recast as a dictionary
    object. Returns numpy array of gc content.
    
    Parameters
    ----------
    fasta_dict: dict
        Dictionary of sequences.

    Returns
    -------
    numpy.ndarray  
        Array of GC percentages.
    """
    gc_c = np.empty(len(fasta_dict))
    atcg_c = np.empty(len(fasta_dict))
    for num, key in enumerate(fasta_dict):
        gc_c[num] = fasta_dict[key].count('G') + fasta_dict[key].count('C')
        atcg_c[num] = gc_c[num] + fasta_dict[key].count('A') + fasta_dict[key].count('T')
    return gc_c.sum() / atcg_c.sum()
        
def get_seq_lengths(fasta_dict):
    """
    Get lengths of sequences in a SeqIO parsed 
    and fasta file recast as a dictionary object.
    Returns numpy array of lengths.
    
    Parameters
    ----------
    fasta_dict : dict
        Dictionary of sequences.
    
    Returns
    -------
    numpy.ndarray
        Array of sequence lengths.
    """
    lengths = []
    for key in fasta_dict:
        lengths.append(len(fasta_dict[key]))
    return np.array(lengths) 


def main(fasta_file):
    """
    Calculate mean N and GC percentages and sequence lengths
    and output to a pandas dataframe.
    1. check if mag.stat.csv exists if it is open it and append
    2. if it doesn't exist create it and write header
    3. process fasta files for stats and append to mag.stat.csv
       if the fasta file is already in mag.stat.csv skip it
    """
    in_files = os.listdir(fasta_file)
    if os.path.isfile('mag.stat.csv'):
        df = pd.read_csv('mag.stat.csv', index_col=0)
    else:
        df = pd.DataFrame(columns=['N_perc', 'GC_perc', 'seq_length'])
    for in_file in in_files:
        if in_file not in df.index:
            fasta_dict = parser(fasta_file + "/" + in_file)
            n_perc = get_n_perc(fasta_dict)
            gc_perc = get_gc_perc(fasta_dict)
            seq_lengths = get_seq_lengths(fasta_dict)
            df.loc[in_file.strip(".fna")] = [n_perc.mean(), gc_perc, seq_lengths.mean()]
    df.to_csv("mag_stats.csv")
             

if __name__ == '__main__':
    args = arg_parser()
    main(args.fasta_repo)
