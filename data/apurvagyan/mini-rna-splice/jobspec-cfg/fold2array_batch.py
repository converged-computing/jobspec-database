from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count, Lock
import pickle
import os
import json

import sys
sys.path.append('../GSAE/gsae/data_processing')
from utils import dot2adj

# Function to convert adjacency matrix to adjacency list
def adj_matrix_to_adj_list(adj_matrix):
    num_nodes = len(adj_matrix)
    adj_list = {}

    for i in range(num_nodes):
        adj_list[i] = [j for j in range(num_nodes) if adj_matrix[i][j] != 0]

    return adj_list

# Function to process each record
def process_record(record):
    event_id = record[0].split(': ')[1].strip()
    sequence = record[1].split(': ')[1].strip()
    secondary_structure = record[2].split(': ')[1].strip()
    min_free_energy = float(record[3].split(': ')[1].strip())
    
    adj_mat = dot2adj(secondary_structure)
    adj_list = adj_matrix_to_adj_list(adj_mat)
    
    return event_id, sequence, min_free_energy, adj_list

def chunks(iterable, chunk_size):
    """Yield successive chunks of chunk_size from iterable, discarding the fourth line."""
    for i in range(0, len(iterable), chunk_size):
        chunk = iterable[i:i + chunk_size]
        # Discard the fourth line
        yield chunk[:chunk_size-1]

def write_data(results, lock, fnames):
    for data in results:
        event_id, sequence, min_free_energy, adj_list = data

        lock.acquire()

        with open(fnames[0], 'a') as file:
            file.write(sequence)
            file.write('\n')

        with open(fnames[1], 'ab') as file:
            pickle.dump(adj_list, file)

        # print(adj_list)
        # with open('output_adjacency.json', 'a') as file:
        #     json.dump(adj_list, file)
        #     file.write('\n')  # Append newline character

        with open(fnames[2], 'a') as file:
            file.write(str(min_free_energy))
            file.write('\n')

        with open(fnames[3], 'a') as file:
            file.write(event_id)
            file.write('\n')

        lock.release()

def main():


    input_filename = 'folded_240507_5000.txt'
    output_filename = 'outputs_240507_5000'

    # input_filename = 'test_folded.txt'
    # output_filename = 'test_outputs'

    seq_fname = "{}_sequences.txt".format(output_filename)
    adj_list_fname = "{}_adjacency.pkl".format(output_filename)
    energy_fname = "{}_energies.txt".format(output_filename)
    event_id_fname = "{}_events.txt".format(output_filename)

    fnames = [seq_fname, adj_list_fname, energy_fname, event_id_fname]

    for fname in fnames:
        if os.path.exists(fname):
            os.remove(fname)
    
    lock = Lock()

    # Open the file
    with open(input_filename, 'r') as file:
        # Read the file line by line
        lines = file.readlines()

        fold_data = chunks(lines, 5)

        # fold_data = [next(fold_data) for _ in range(4)]        

        with Pool(processes=cpu_count()) as pool:
            
            results = tqdm(pool.imap(process_record, fold_data), total=len(lines) // 4)

            write_data(results, lock, fnames)
            
    # Extract results
    # sequences, min_free_energies, adj_lists = [np.array(item) for item in np.array(results).T]

    # # Save results
    # np.savetxt(seq_fname, sequences, delimiter='\n', fmt='%s')
    # np.savetxt(energy_fname, min_free_energies, delimiter='\n',fmt='%s')
    # np.save(adj_list_fname, adj_lists)

if __name__ == "__main__":
    main()
