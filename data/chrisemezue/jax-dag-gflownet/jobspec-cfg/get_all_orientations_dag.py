import os
import sys
import numpy as np
import causaldag as cd
import networkx as nx
import pickle as pl
from tqdm import tqdm

# The plan is to get the Markov equivalence class of all ground-truth DAGs (graphs).


# Source: https://github.com/ServiceNow/typed-dag/blob/e951e7323f15b7b16499d5690e7105bd4d3956f0/typed_pc/tmec.py#L18
def is_acyclic(adjacency: np.ndarray) -> bool:
    """
    Check if adjacency matrix is acyclic
    :param adjacency: adjacency matrix
    :returns: True if acyclic
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True

if __name__=="__main__":
    BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20'



    #baseline_to_use = ['bcdnets','bootstrap_ges','bootstrap_pc','dibs','gadget','mc3','dag-gfn']
    baseline_to_use = ['dag-gfn']

    SEED_TO_USE = [i for i in range(26)]
    variance=[]
    with tqdm(total = len(baseline_to_use)*len(SEED_TO_USE)) as pbar:
        for baseline in baseline_to_use:
            for seed in SEED_TO_USE:
                baseline_path = os.path.join(BASELINE_FOLDER,baseline)
                BASE_PATH = os.path.join(baseline_path,str(seed))

                if not os.path.exists(os.path.join(BASE_PATH,'true_mec_dags.pkl')):

                    if not os.path.exists(BASE_PATH):
                        continue

                    cpdag_filepath = os.path.join(BASE_PATH,'true_cpdag.pkl')
                    
                    if not os.path.exists(cpdag_filepath):
                        print(f"could not find cpdag filepath: {cpdag_filepath}")
                        continue        
                    
                    true_mec_dags_filename = os.path.join(BASE_PATH,'true_mec_dags.pkl')
                    if not os.path.exists(true_mec_dags_filename):
                        with open(cpdag_filepath,'rb') as f:
                            cpdag = pl.load(f)
                        
                        MEC_DAGS=[]
                        # Now get all possible orientations
                        for d in cd.PDAG.from_amat(cpdag).all_dags():
                            adj_d = cd.DAG(nodes=np.arange(cpdag.shape[0]), arcs=d).to_amat()[0]
                            if is_acyclic(adj_d):
                                MEC_DAGS.append(adj_d)
                        with open(true_mec_dags_filename,'wb+') as f:
                            pl.dump(MEC_DAGS,f)
                pbar.update(1)

    print('ALL DONE')
