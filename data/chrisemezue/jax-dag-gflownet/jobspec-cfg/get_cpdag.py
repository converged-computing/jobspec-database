import os
import sys
import cdt
import networkx as nx
import pickle as pl
from tqdm import tqdm
from cdt.metrics import get_CPDAG


# The plan is to get the Markov equivalence class of all ground-truth DAGs (graphs).

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

                if not os.path.exists(BASE_PATH):
                    continue
                true_cpdag_filename = os.path.join(BASE_PATH,'true_cpdag.pkl')
                if not os.path.exists(true_cpdag_filename):
                    with open(os.path.join(BASE_PATH,'graph.pkl'),'rb') as fl:
                        graph = pl.load(fl)
                    graph = nx.DiGraph(graph)
                    
                    cpdag = get_CPDAG(graph)

                    with open(true_cpdag_filename,'wb+') as f:
                        pl.dump(cpdag,f)
                pbar.update(1)

    print('ALL DONE')
