import argparse
from sklearn.svm import SVC
from tool import _adata_processing
import json
import scanpy as sc
import os

def main(full_dataset_1000var_pca_path ,refined_cluster_path, predicted_output_directory):

    full_dataset_1000var_pca = sc.read_h5ad(full_dataset_1000var_pca_path)
    dataset_200_1000var_pca = _adata_processing.subset_anndata_from_cluster_dictionary(full_dataset_1000var_pca, refined_cluster_path)

    with open(refined_cluster_path) as json_file:
        refined_cluster_200 = json.load(json_file)
    

    svm_100 = SVC(kernel='linear')

    print(dataset_200_1000var_pca)
    print(refined_cluster_200.values())
    svm_100.fit(dataset_200_1000var_pca.obsm['X_pca'][:, :10], list(refined_cluster_200.values()))
    

    mali_pred = svm_100.predict(full_dataset_1000var_pca.obsm['X_pca'][:, :10])
    
    predicted_clusters = create_cluster_dictionary(full_dataset_1000var_pca, mali_pred)


    ################### output & filename generation
    # Ensure the output directory exists

    # Extract the basename and split it
    initial_cluster_basename = os.path.basename(refined_cluster_path)
    parts = initial_cluster_basename.split('_')
    
    # Replace the first element (prefix) with 'refined'
    parts[0] = 'predicted'
    parts[1] = 'full'
    
    # Construct the new filename
    refined_filename = '_'.join(parts)
    
    # Construct the full file path for the output
    file_path = os.path.join(predicted_output_directory, refined_filename)

    # Save the refined cluster data to the new file
    with open(file_path, 'w') as json_file:
        json.dump(predicted_clusters, json_file, indent=4)
    
    # Provide feedback to the user or for logging
    print(f"Predicted cluster data saved to {file_path}")



def create_cluster_dictionary(adata, predictions):
    """
    Create a dictionary with cell barcodes as keys and cluster assignments as values.
    
    Parameters:
    adata: AnnData object
    predictions: list or array of cluster assignments corresponding to the cells in `adata`
    
    Returns:
    A dictionary of the format {barcode: cluster}
    """
    # Ensure the number of predictions matches the number of cells in adata
    assert adata.n_obs == len(predictions), "Number of predictions must match number of cells in adata."
    
    # Extract the barcodes
    barcodes = adata.obs.index.tolist()
    
    # Create the dictionary
    cluster_dict = {barcode: str(cluster) for barcode, cluster in zip(barcodes, predictions)}
    
    return cluster_dict



#USAGE: python predict_clusters.py --full_dataset path_to_full_dataset.h5ad --refined_cluster path_to_refined_cluster.json --output_dir path_to_output_directory
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SVM prediction and save cluster dictionary.')
    parser.add_argument('-f', '--full_dataset', type=str, required=True,
                        help='Path to full dataset h5ad file.')
    parser.add_argument('-r', '--refined_cluster', type=str, required=True,
                        help='Path to the refined cluster JSON file.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save predicted cluster dictionary.')
    
    args = parser.parse_args()
    
    main(args.full_dataset, args.refined_cluster, args.output_dir)