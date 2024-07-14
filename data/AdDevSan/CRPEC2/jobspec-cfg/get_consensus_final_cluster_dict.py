import numpy as np
import json

from glob import glob

import os
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

import argparse

'''
Takes in sample directory, identifies predicted_clusters subdirectory. 

Outputs consensus matrix csv and final clusters cluster dict json (ensemble cluster).


'''

def load_cluster_jsons(directory):
    """Loads all predicted cluster JSONs and returns a list of dictionaries."""
    cluster_dictionaries_list = []
    barcodes_list = []
    for filename in os.listdir(directory):
        if filename.startswith('predicted_full_') and filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:

                cluster_dictionary = json.load(file)
                cluster_dictionaries_list.append(cluster_dictionary)
        

    #simply takes keys of first dictionary in cluster_dictionaries_list with the assumption that all cluster dictionary keys are same
    barcodes_list = list(cluster_dictionaries_list[0].keys())

    return cluster_dictionaries_list, barcodes_list

def convert_dicts_to_arrays(cluster_dictionaries_list):
    """Converts a list of clustering dictionaries into a 2D NumPy array."""
    num_iterations = len(cluster_dictionaries_list)
    all_barcodes = sorted(cluster_dictionaries_list[0].keys())
    num_cells = len(all_barcodes)

    # Initialize the clustering array
    clustering_array = np.zeros((num_iterations, num_cells), dtype=int)

    # Map the cluster labels to integers for each iteration independently
    for iteration_idx, clustering in enumerate(cluster_dictionaries_list):
        # Get unique clusters for this iteration and sort them
        unique_clusters = sorted(set(clustering.values()))
        # Map clusters to integers
        cluster_to_int = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
        # Fill the array for this iteration using the map
        clustering_array[iteration_idx] = [cluster_to_int[clustering[barcode]] for barcode in all_barcodes]

    return clustering_array

def compute_consensus_matrix(clusterings_matrix, num_iterations):
    """Computes the consensus matrix from a 2D NumPy array of cluster assignments."""

    comparison_matrix = clusterings_matrix[:, :, None] == clusterings_matrix[:, None, :]
    consensus_matrix = comparison_matrix.sum(axis=0) / num_iterations
    return consensus_matrix



def get_linkage_distance_matrix(consensus_matrix):
        # Convert the consensus matrix into a condensed distance matrix
    # This assumes consensus_matrix is a similarity matrix; adjust if it's already a distance matrix
    max_similarity = np.max(consensus_matrix)
    distance_matrix = max_similarity - consensus_matrix
    condensed_distance_matrix = squareform(distance_matrix)
    

    # Perform hierarchical clustering to get linkage matrix
    linkage_matrix = linkage(condensed_distance_matrix, method='ward')

    return linkage_matrix, distance_matrix
    
def determine_optimal_clusters(linkage_matrix, distance_matrix, max_cluster_range):
    """
    Determine the optimal number of clusters by computing silhouette scores for different cluster counts.

    Parameters:
    consensus_matrix: numpy.ndarray
        A square matrix of consensus scores between objects.
    max_cluster_range: int
        The maximum number of clusters to consider.

    Returns:
    best_num_clusters: int
        The optimal number of clusters with the highest silhouette score.
    best_sil_score: float
        The best silhouette score achieved.
    """



    # Initialize the best score and best cluster count
    best_num_clusters = 0
    best_sil_score = -1

    # Test cluster counts from 2 up to the specified maximum range
    for n_clusters in range(2, max_cluster_range + 1):
        # Extract cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate the silhouette score (using the non-condensed distance matrix)
        sil_score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        
        # Update the best score and cluster count if the current score is better
        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_num_clusters = n_clusters

    # Return the best cluster count and the corresponding silhouette score
    return best_num_clusters, best_sil_score

# Example usage:
# consensus_matrix = ... (your precomputed consensus matrix here)
# max_cluster_range = 10
# best_clusters, best_score = determine_optimal_clusters(consensus_matrix, max_cluster_range)
# print(f"Best number of clusters: {best_clusters} with a silhouette score of: {best_score}")



def extract_cluster_labels(linkage_matrix, best_num_clusters, barcodes_list):
    """
    Assigns cluster labels based on the optimal number of clusters determined by silhouette analysis.

    Parameters:
    linkage_matrix: ndarray
        The linkage matrix obtained from hierarchical clustering.
    best_num_clusters: int
        The optimal number of clusters determined by silhouette analysis.

    Returns:
    dict
        A dictionary mapping each cell index to its cluster label.
    """
    # Assign cluster labels using the optimal number of clusters determined
    cluster_labels = fcluster(linkage_matrix, best_num_clusters, criterion='maxclust')
    
    # Create a dictionary to map cell indices to their corresponding cluster labels, converting NumPy types to Python types
    # ZIP ENSURES barcodes_list length matches cluster_labels
    cell_cluster_mapping = {barcode: str(label) for barcode, label in zip(barcodes_list,cluster_labels)}
    
    return cell_cluster_mapping



def main(args):

    # Your existing code to process the directory
    sample_directory_path = args.directory
    predicted_clusters_directory = os.path.join(sample_directory_path, "predicted_clusters")
    cluster_dictionaries_list, barcodes_list = load_cluster_jsons(predicted_clusters_directory)


    # Convert the cluster dictionaries to arrays
    clustering_array = convert_dicts_to_arrays(cluster_dictionaries_list)



    num_iterations = clustering_array.shape[0]
    consensus_matrix = compute_consensus_matrix(clustering_array, num_iterations)


    np.savetxt(os.path.join(sample_directory_path, "consensus_matrix.csv"), consensus_matrix, delimiter=",")

    
    consensus_matrix = np.loadtxt(os.path.join(sample_directory_path, "consensus_matrix.csv"), delimiter=',')

    max_cluster_range = 10
    
    #LINKAGE MATRIX
    linkage_matrix, distance_matrix= get_linkage_distance_matrix(consensus_matrix)

    #SILHOUETTE
    best_num_clusters, best_score = determine_optimal_clusters(linkage_matrix, distance_matrix, max_cluster_range)
    print(f"Best number of clusters: {best_num_clusters} with a silhouette score of: {best_score}")



    # Sample usage of the function
    # Assuming linkage_matrix is already defined and best_num_clusters has been determined
    # Get the cluster mapping
    
    cell_cluster_mapping = extract_cluster_labels(linkage_matrix, best_num_clusters, barcodes_list)

    # Save the cluster mapping to a JSON file
    with open(os.path.join(sample_directory_path,'ensemble_cluster_full.json'), 'w') as f:
        json.dump(cell_cluster_mapping, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cluster data.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="The path to the sample directory containing clustering data.")
    args = parser.parse_args()
    main(args)