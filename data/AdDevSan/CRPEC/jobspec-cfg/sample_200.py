import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def sample_and_output(barcodes_df, sample_size, output_dir_samples, output_dir_remainders, iteration):
    # Sample `sample_size` barcodes and split into two dataframes
    sampled_barcodes, remainder_barcodes = train_test_split(barcodes_df, train_size=sample_size, shuffle=True)

    # Create output directories if they don't exist
    os.makedirs(output_dir_samples, exist_ok=True)
    os.makedirs(output_dir_remainders, exist_ok=True)

    # Write the sampled barcodes to a file
    sampled_barcodes.to_csv(os.path.join(output_dir_samples, f'sample_200_{iteration}.tsv'), header=False, index=False, sep='\t')

    # Write the remaining barcodes to a file
    remainder_barcodes.to_csv(os.path.join(output_dir_remainders, f'sample_remainder_{iteration}.tsv'), header=False, index=False, sep='\t')

def main(barcodes_file, n_samples, output_dir_samples, output_dir_remainders, sample_size=200):
    # Read the barcodes file
    barcodes_df = pd.read_csv(barcodes_file, header=0, sep='\t')

    for i in range(n_samples):
        sample_and_output(barcodes_df, sample_size, output_dir_samples, output_dir_remainders, i+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample N barcodes from a TSV file multiple times and output to respective directories.")

    parser.add_argument('-b', '--barcodes_tsv', type=str, required=True, help='Path to the barcodes TSV file to sample from.')
    parser.add_argument('-n', '--n_samples', type=int, required=True, help='Number of times to perform the sampling.')
    parser.add_argument('-s', '--output_dir_samples', type=str, required=True, help='Output directory for the 200 samples.')
    parser.add_argument('-r', '--output_dir_remainders', type=str, required=True, help='Output directory for the remainders.')

    args = parser.parse_args()

    main(args.barcodes_tsv, args.n_samples, args.output_dir_samples, args.output_dir_remainders)
