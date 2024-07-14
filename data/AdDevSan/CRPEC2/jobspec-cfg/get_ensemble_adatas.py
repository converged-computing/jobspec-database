import scanpy as sc
import json
import pandas as pd
import os
import argparse
from tool import _adata_processing
def get_subset_cluster_dict_list(cluster_dict):
    unique_clusters = list(set(cluster_dict.values()))

    subset_cluster_dict_list = []
    for cluster in unique_clusters:
        # Create a subset dictionary where each key-value pair matches the current cluster
        subset_dict = {key: value for key, value in cluster_dict.items() if value == cluster}
        # Add this subset dictionary to the list
        subset_cluster_dict_list.append(subset_dict)

    return subset_cluster_dict_list
def get_gene_profiles(adata_input, top_n):
    adata = None
    adata = adata_input.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=top_n, subset=True)
    return list(adata.var.index)




def main(sample_name, ensemble_cluster_path, processed_adata_path, output_directory_path):


    with open(ensemble_cluster_path) as json_file:
        ensemble_cluster_dict = json.load(json_file)

    processed_adata = sc.read_h5ad(processed_adata_path)
    #processed_adata.var['highly_variable'] = None
    subset_cluster_dict_list = get_subset_cluster_dict_list(ensemble_cluster_dict)

    print("processed", processed_adata)
    for index, cluster_dict in enumerate(subset_cluster_dict_list):

        ensemble_adata = None
        


        ensemble_adata = _adata_processing.subset_anndata_from_cluster_dictionary2(processed_adata, cluster_dict)
        #print("leak test:", ensemble_adata)
        
        #ensemble_adata.var['highly_variable'] = None
          # Clear existing annotations if any


        cluster_value = str(list(cluster_dict.values())[0])
        
        print(cluster_value)
        print(ensemble_adata)
       

        filename = f"ensemble_adata_{sample_name}_{cluster_value}.h5ad"
        ensemble_adata.write(os.path.join(output_directory_path, filename))
        
        '''

        barcodes = list(cluster_dict.keys())
        mask = processed_adata.obs.index.isin(barcodes)
        adata_subset = processed_adata[mask]

        subset_gene_profile_list = get_gene_profiles(adata_subset, 200)

        

        # Assuming subset_gene_profile_list is a list of tuples/lists where each tuple/list represents a gene and its profile
        # Convert the list to a DataFrame
        gene_profile_df = pd.DataFrame(subset_gene_profile_list)

        # Define the output path for the TSV file
        
        output_path = os.path.join(output_directory_path, filename) 

        # Save the DataFrame to a TSV file
        gene_profile_df.to_csv(output_path, sep='\t', index=False, header=False)'''




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract top variable genes from single cell clusters.")
    parser.add_argument("-sn", "--sample_name", type=str, required=True, help="Sample name identifier.")
    parser.add_argument("-ecp", "--ensemble_cluster_path", type=str, required=True, help="Path to ensemble cluster JSON.")
    parser.add_argument("-pp", "--processed_adata_path", type=str, required=True, help="Path to the processed h5ad file.")
    parser.add_argument("-odp", "--output_directory_path", type=str, required=True, help="Directory path to save the cluster adatas.")

    args = parser.parse_args()

    main(
        args.sample_name,
        args.ensemble_cluster_path,
        args.processed_adata_path,
        args.output_directory_path
    )

'''
    sample_name = "GSM4909299"
    ensemble_cluster_path = r"runs\CRPEC_run_20240414_031735\GSM4909299\ensemble_cluster_full.json"
    processed_adata_path = r"runs\CRPEC_run_20240414_031735\GSM4909299\full_dataset_processed.h5ad"
    output_directory_path = r'runs/CRPEC_run_20240414_031735/GSM4909299/gene_profiles'
'''

'''
python3 get_gene_profiles.py \
    --sample_name "GSM4909299" \
    --ensemble_cluster_path "runs/CRPEC_run_20240414_031735/GSM4909299/ensemble_cluster_full.json" \
    --processed_adata_path "runs/CRPEC_run_20240414_031735/GSM4909299/full_dataset_processed.h5ad" \
    --output_directory_path "runs/CRPEC_run_20240414_031735/GSM4909299/gene_profiles"
'''