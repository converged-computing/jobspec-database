import os
import argparse
import scanpy as sc
from tool import _adata_processing

def main(trident_directory, h5ad_directory, filtered_barcodes_directory):
    adata = _adata_processing.get_adata_from_trident(trident_directory_path=trident_directory)
    print("adata", adata)
    adata_processed = _adata_processing.preprocess_adata(adata)
    print(adata_processed)
    # Extract the base name for the output files from the trident directory path
    base_name = os.path.basename(trident_directory)

    # Write the processed AnnData object to the h5ad file
    h5ad_file_path = os.path.join(h5ad_directory, f"full_dataset_processed.h5ad")
    adata_processed.write(h5ad_file_path)
    print(f"Processed H5AD file saved to: {h5ad_file_path}")
    
    # Write the filtered barcodes to a TSV file
    filtered_barcodes_file_path = os.path.join(filtered_barcodes_directory, f"filtered_barcodes.tsv")
    adata_processed.obs.to_csv(filtered_barcodes_file_path, sep='\t', header=True, index=True)
    print(f"Filtered barcodes file saved to: {filtered_barcodes_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Trident data and output H5AD and barcodes TSV.")
    
    # Trident directory argument
    parser.add_argument('-t', '--trident_directory', required=True, help='The directory containing the Trident files.')
    
    # H5AD output directory argument
    parser.add_argument('-hd', '--h5ad_directory', required=True, help='The directory to save the output H5AD file.')
    
    # Filtered barcodes directory argument
    parser.add_argument('-bd', '--filtered_barcodes_directory', required=True, help='The directory to save the filtered barcodes TSV file.')

    args = parser.parse_args()

    main(args.trident_directory, args.h5ad_directory, args.filtered_barcodes_directory)
