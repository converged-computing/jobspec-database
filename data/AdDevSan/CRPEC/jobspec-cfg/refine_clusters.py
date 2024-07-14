
import os
import glob
import pandas as pd
import scanpy as sc
from scipy.io import mmread
import numpy as np
from sklearn.svm import SVC
import os
import glob
import numpy as np
from sklearn.model_selection import StratifiedKFold, permutation_test_score

from sklearn.svm import SVC
from itertools import combinations
import argparse
from tool import _adata_processing
import json

def trainSVM10CV(principal_components, cluster_information):

    svm_classifier = SVC(kernel='linear')  
    svm_classifier.fit(principal_components, cluster_information)

    return svm_classifier




def getClusterPairCombinations(cluster_dict):
    # Get unique clusters
    unique_clusters = set(cluster_dict.values())
    
    # Generate all unique pairs of clusters
    cluster_pair_combinations = list(combinations(unique_clusters, 2))

    return cluster_pair_combinations

def getShuffleLists(adata, cluster_dict, cluster_pair_combinations):
    

    def convertClusterBinary(cluster_dict):
        unique_values = list(set(cluster_dict.values()))
        value_to_label = {unique_values[0]: 0, unique_values[1]: 1}
        return [value_to_label[value] for value in cluster_dict_subset.values()]
    
    pc_list = []
    label_list = []

    for cluster_pair in cluster_pair_combinations:

        cluster_dict_subset = {key: value for key, value in cluster_dict.items() if value in set(cluster_pair)}
        label = convertClusterBinary(cluster_dict_subset)
        
        
        barcodes = list(cluster_dict_subset.keys())
        
        mask = adata.obs.index.isin(barcodes)
        adata_subset = adata[mask]

        #the steps above should have ensured adata_subset rows match that of cluster_dict_subset
        ten_principal_components_adata_subset = adata_subset.obsm['X_pca'][:, :10]
    
        label_list.append(label)
        pc_list.append(ten_principal_components_adata_subset)

    return pc_list, label_list



def permTestPVal(pc_list, label_list):

    p_value_list = []


    for idx, pc in enumerate(pc_list):
        print("label", label_list[idx])
        print("pc", len(pc))
        perm_svm = SVC(kernel='rbf')
        cv = StratifiedKFold(n_splits=10)
        try:
            score, permutation_scores, p_value = permutation_test_score(perm_svm, X= pc, y = label_list[idx], scoring = "accuracy", cv = cv, n_permutations= 100, n_jobs= -1)
        except ValueError as n_splits_ve:
            print("n_splits too big, returning p_value = 1")
            p_value = 1
        print("p_value", p_value)
        p_value_list.append(p_value)

    return p_value_list


def mergeClusterPair(cluster_dictionary, cluster_pair):
    new_cluster_name = f"{cluster_pair[0]}n{cluster_pair[1]}"

    # Iterate over the cluster_dictionary items
    for key, value in cluster_dictionary.items():
        # Check if the value matches any of the items in cluster_pair
        if value in cluster_pair:
            # Update the value to the new cluster name
            cluster_dictionary[key] = new_cluster_name
            
    # Return the updated dictionary
    return cluster_dictionary




def refineClusters(adata, cluster_dictionary): 



    cluster_pair_combinations = getClusterPairCombinations(cluster_dictionary) #a
    pc_list, label_list = getShuffleLists(adata, cluster_dictionary, cluster_pair_combinations) #b pc_list: list of PC df-like; label_list: list of dicts

    print("pc_list", [len(i) for i in pc_list])
    print("label_list", [len(i) for i in label_list])
    p_val_list = permTestPVal(pc_list, label_list) #c

    
    print("p_val_list", p_val_list)
    top_p = max(p_val_list)
    
    #BASE CASE
    if top_p < 0.05 or len(p_val_list) <= 2:
        return cluster_dictionary
    
    top_p_index = p_val_list.index(top_p)
    cluster_pair_to_merge = cluster_pair_combinations[top_p_index]

    refined_cluster = mergeClusterPair(cluster_dictionary, cluster_pair=cluster_pair_to_merge)

    return refineClusters(adata, refined_cluster)



    



#misc functions
def file_exists(file_path):
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"{file_path} does not exist or is not a file.")
    return file_path

def dir_exists(directory_path):
    if not os.path.isdir(directory_path):
        raise argparse.ArgumentTypeError(f"{directory_path} does not exist or is not a directory.")
    return directory_path


#MAIN



def main(h5ad_file_path, initial_cluster_json, output_directory="refined_clusters"):


    adata_full = sc.read_h5ad(h5ad_file_path)
    adata_200 = _adata_processing.subset_anndata_from_cluster_dictionary(adata_full, initial_cluster_json)
    with open(initial_cluster_json) as json_file:
        initial_cluster_dict = json.load(json_file)

    #GET 1000 TOP GENES & PCA
    sc.pp.highly_variable_genes(adata_200, n_top_genes=1000, subset=True)
    sc.tl.pca(adata_200, svd_solver='arpack')

    # REFINE CLUSTERS
    refined_cluster = refineClusters(adata_200, initial_cluster_dict)

    # Ensure the output directory exists

    # Extract the basename and split it
    initial_cluster_basename = os.path.basename(initial_cluster_json)
    parts = initial_cluster_basename.split('_')
    
    # Replace the first element (prefix) with 'refined'
    parts[0] = 'refined'
    
    # Construct the new filename
    refined_filename = '_'.join(parts)
    
    # Construct the full file path for the output
    file_path = os.path.join(output_directory, refined_filename)

    # Save the refined cluster data to the new file
    with open(file_path, 'w') as json_file:
        json.dump(refined_cluster, json_file, indent=4)
    
    # Provide feedback to the user or for logging
    print(f"Refined cluster data saved to {file_path}")
'''    os.makedirs(output_directory, exist_ok=True)


    #TODO: change file_path to take initial_cluster_json and replace word before first _ to refined
    #TODO: example: initial_200_1.json -> refined_200_1.json
    file_path = os.path.join(output_directory, f"refined_cluster_{iteration}.json")
    with open(file_path, 'w') as json_file:
        json.dump(refined_cluster, json_file, indent=4)'''




#EXAMPLE USE CASE: TAKES H5AD
#python refine_clusters.py -f path/to/data.h5ad -b path/to/barcodes_sample_200.tsv -i path/to/initial_cluster.json -o path/to/output_directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process single-cell sequencing data with subsetting and cluster refinement.")

    # H5AD file argument
    parser.add_argument('-f', '--h5ad_file_path', required=True, type=file_exists, help='Path to the .h5ad file.')



    # Initial cluster JSON file argument
    parser.add_argument('-i', '--initial_cluster_json', required=True, type=file_exists, help='Path to the initial cluster JSON file.')

    # Optional output directory argument
    parser.add_argument('-o', '--output_directory', default="refined_clusters", type=str, help='The output directory for saving refined clusters JSON file.')

    args = parser.parse_args()

    main(args.h5ad_file_path, args.initial_cluster_json, args.output_directory)

