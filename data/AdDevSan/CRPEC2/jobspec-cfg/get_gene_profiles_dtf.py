import scanpy as sc
import os
import glob
import pandas as pd
import numpy as np
from scipy.sparse import issparse
import argparse


def get_ensemble_adatas_path_dict(run_id):
    ensemble_adatas_dict = {}
    
    # List all sample directories
    sample_dirs = glob.glob(os.path.join(run_id, 'GSM*'))
    
    for sample_dir in sample_dirs:
        # Extract the sample name from the path (last part of the path)
        sample_name = os.path.basename(sample_dir)
        
        # Path to ensemble_adatas directory
        ensemble_adatas_dir = os.path.join(sample_dir, 'ensemble_adatas')
        
        # List all h5ad files in the ensemble_adatas directory
        h5ad_files = glob.glob(os.path.join(ensemble_adatas_dir, '*.h5ad'))
        
        # Store the list of h5ad file paths in the dictionary under the sample name key, formatted with forward slashes
        ensemble_adatas_dict[sample_name] = [os.path.normpath(path).replace(os.sep, '/') for path in h5ad_files]
    
    return ensemble_adatas_dict



def get_unique_genes_list(adatas_path_dict):
    
    big_genes_list = []
    adatas_path_list = [item for sublist in list(adatas_path_dict.values()) for item in sublist]

    for adata_path in list(adatas_path_list):
        adata = None
        adata = sc.read_h5ad(adata_path)
        print(adata)
        sc.pp.highly_variable_genes(adata=adata, n_top_genes=200, subset=True)
        big_genes_list += list(adata.var.index)
        print(len(big_genes_list), len(adata.var.index))

    return list(set(big_genes_list))


import scanpy as sc

def subset_adata_by_genes(adata, gene_list):
    # Filter the unique_genes_list to include only those that are present in the adata's var_names
    genes_present = [gene for gene in gene_list if gene in adata.var_names]
    # Subset the adata to only include the genes present
    adata_subset = adata[:, genes_present]
    return adata_subset


def get_mean_expression(adata):
    if issparse(adata.X):
        # Convert the sparse matrix to a dense array to calculate the median
        median_expression = np.mean(adata.X.toarray(), axis=0)
    else:
        # Calculate the median expression across cells for each gene
        median_expression = np.mean(adata.X, axis=0)
    
    gene_expression_dict = dict(zip(adata.var_names, median_expression))
    return gene_expression_dict
    

def get_gene_profiles_dtf(ensemble_adatas, unique_genes_list):
    # Initialize dataframe with columns corresponding to genes of unique_genes_list
    gene_profiles_dtf = pd.DataFrame(columns=unique_genes_list)
    
    for sample, ensemble_adata_list in ensemble_adatas.items():  # use .items() for key, value iteration
        for adata_path in ensemble_adata_list:
            adata = None
            adata = sc.read_h5ad(adata_path)
            
            # Subset adata based on unique_genes_list
            adata_ug_subset = subset_adata_by_genes(adata, unique_genes_list)
            
            print("adata_subset", adata_ug_subset)
            # Get median expression of each gene expression among barcodes
            gene_profile_dict_median = get_mean_expression(adata_ug_subset)
            print(gene_profile_dict_median)
            # Create a Series from the dictionary, ensuring it aligns with the DataFrame's columns
            median_expression_series = pd.Series(gene_profile_dict_median, index=gene_profiles_dtf.columns)
            median_expression_series.fillna(0, inplace=True)
            # Extract cluster information from the file path
            # Assuming the path format is consistent and cluster information is right before the file extension
            cluster = os.path.splitext(os.path.basename(adata_path))[0].split('_')[-1]
            
            # Add row to DataFrame, naming it using sample and cluster
            row_name = f"{sample}_{cluster}"
            gene_profiles_dtf.loc[row_name] = median_expression_series

    return gene_profiles_dtf



# Define the previously mentioned functions here...

def main(run_id, output_file):
    ensemble_adatas = get_ensemble_adatas_path_dict(run_id)
    unique_genes_list = get_unique_genes_list(ensemble_adatas)
    gene_profiles_dataframe = get_gene_profiles_dtf(ensemble_adatas, unique_genes_list)
    gene_profiles_dataframe.to_csv(output_file, index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ensemble adata files and generate a gene profiles dataframe.")
    parser.add_argument("--run_id", type=str, help="The run ID directory to process", required=True)
    parser.add_argument("--output_file", type=str, help="Output CSV file path", required=True)
    args = parser.parse_args()

    main(args.run_id, args.output_file)