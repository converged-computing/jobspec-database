import pickle as pk
import numpy as np
from scatter import Scatter
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import psutil
from multiprocessing import Pool, Lock, cpu_count, Manager, Queue
from functools import partial


def adj_list_to_adj_matrix(adj_list):
    """
    Convert an adjacency list to an adjacency matrix.
    
    Parameters:
        adj_list (dict): The adjacency list representing the graph.
    
    Returns:
        numpy.ndarray: The adjacency matrix.
    """
    num_nodes = len(adj_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1  # or any other value to represent the edge weight

    return adj_matrix
def adjacency_matrix_to_edge_index(adjacency_matrix):
    """
    Convert adjacency matrix to edge index (COO format).
    """
    coo = coo_matrix(adjacency_matrix)
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    return edge_index


def get_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    return mem

def write_data(results, lock, fnames):
    
    graph_data = results

    lock.acquire()

    with open(fnames, 'ab') as file:
        pk.dump(graph_data, file)

    # with open(fnames[1], 'a') as file:
    #     file.write(gene_id + '\n')

    lock.release()


def process_data(data, out_fnames, write_lock):

    matrix = adj_list_to_adj_matrix(data)
    edge_index = adjacency_matrix_to_edge_index(matrix)
    num_nodes = len(data)
    x = torch.eye(num_nodes)
    graph_data = Data(x=x, edge_index=edge_index)

    write_data(graph_data, write_lock, out_fnames)
    

def main():
    # unique_gene_ids = set()
    # with open('unique_gene_ids.txt', 'r') as gene_ids_file:
    #     for line in gene_ids_file:
    #         gene_id = line.strip()
    #         unique_gene_ids.add(gene_id)

    manager = Manager()
    write_lock = manager.Lock()

    # in_fname = '../data/out_success/output_gpu_adjacency.pkl'
    out_fnames = 'graph_data.pkl'


    with Pool(processes=cpu_count()) as pool:
        # with open('../data/out_success/gene_ids.txt', 'r') as gene_ids_file, open(in_fname, 'rb') as in_f:
        #     for line in tqdm(gene_ids_file):
        #         gene_id = line.strip()
        #         data = pk.load(in_f)

        #         if gene_id in unique_gene_ids:
        with open('240507/outputs_240507_2_adjacency.pkl', 'rb') as in_f:
    
            # graph_lst = []
            #Replace the 46925 with exact number of sequences in the file
            for i in (tqdm(range(2465))):
                data = pk.load(in_f)
                # import pdb; pdb.set_trace()
                    
                partial_process_data = partial(process_data, 
                                data=data, 
                                out_fnames=out_fnames, 
                                write_lock=write_lock)

                process_data(data, out_fnames, write_lock)

                    # with tqdm(total=len(unique_gene_ids)) as pbar:
                    #     for _ in pool.imap_unordered(partial_process_data, unique_gene_ids):
                    #         pbar.update(1)

                    # results = list(tqdm(pool.imap_unordered(partial_process_data, range(len(unique_gene_ids))), total=unique_gene_ids))


if __name__ == '__main__':
    main()

                # graph_lst.append(graph_data)
                # gene_ids_lst.append(gene_id)

# Save the graph data
# torch.save(graph_lst, 'graph_data_matched.pt')


# graph_data = torch.load('graph_data_under_200.pt')
# print(len(graph_data))

# # coeffs = []
# for graph in tqdm(graph_data):
#     # print(i)
#     in_channels = graph.x.size(0)
#     max_graph_size = graph.x.size(0)
#     scattering = Scatter(in_channels, max_graph_size)
#     scatter_coeffs = scattering(graph)

#     # coeffs.append(scatter_coeffs)

#     with open('scatter_coeffs_under_200.pkl', 'ab') as out_f:
#         torch.save(scatter_coeffs, out_f)

# import pdb; pdb.set_trace()
# Save the scatter coefficients
# torch.save(coeffs, 'scatter_coeffs_200.pt')
