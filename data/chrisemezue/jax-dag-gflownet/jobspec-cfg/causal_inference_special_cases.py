import os
import sys
import time
import json
import random
import pandas as pd
import numpy as np
from numpy.random import default_rng
import networkx as nx
from dowhy import CausalModel
import pickle as pl
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects
from typing import Optional

set_loky_pickler('pickle5') #https://joblib.readthedocs.io/en/latest/auto_examples/serialization_and_wrappers.html

node_to_index = {
    'A':0,
    'B':1,
    'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19
}

index_to_node = {v:k for k,v in node_to_index.items()} 

baselines_ = []
rmses_ = []
seeds_ = []


BASELINE_FOLDER = '/home/mila/c/chris.emezue/gflownet_sl/tmp/lingauss20'
OUTPUT_FOLDER_NAME = 'ate_estimates_sc'

os.makedirs(f'/home/mila/c/chris.emezue/scratch/causal_inference/{OUTPUT_FOLDER_NAME}',exist_ok=True)
os.makedirs(f'{OUTPUT_FOLDER_NAME}',exist_ok=True)

#BASELINE_FOLDER = '/home/mila/c/chris.emezue/scratch/baselines/'

# New idea: find treatment and outcome variables based on interesting scenarios
# then evaluate the posterior samples of the baselines based on the selected variables.

def get_adj_of_edge(variable,graph):
    # Given a variable, find the set of its children, using the graph
    adj_variables = graph._adj[variable]
    if adj_variables=={}:
        return []
    return list(adj_variables.keys())

def get_pred_of_edge(variable,graph):
    # Given a variable, find the set of its children, using the graph
    adj_variables = graph._pred[variable]
    if adj_variables=={}:
        return []
    return list(adj_variables.keys())


def get_pairwise_elements(list_):
    pairwise = []
    for i in range(len(list_)):
        for j in range(len(list_)):
            if list_[i]!=list_[j]:
                pairwise.append((list_[i],list_[j]))
    return pairwise

def treatment_effect_with_cofounder(graph):
    
    """
    Given a graph, we want to find one treatment and effect satisfying the criterion of having a confounder.
    Where there are many such pairs, we sample one pair randomly.
    
    """
    all_edges = list(graph.edges)
    confounders = [(edge[0],edge[1],set(get_pred_of_edge(edge[0],graph)).intersection(set(get_pred_of_edge(edge[1],graph)))) for edge in all_edges]
    variables_of_interest = [c for c in confounders if len(c[2])!=0]
    variables_of_interest = [(v[0],v[1],c)  for v in variables_of_interest for c in v[2]]
    return variables_of_interest



def treatment_effect_with_cofounder_no_effect(graph):

    """
    Given a graph, we want to find one treatment and effect satisfying the criterion of having a confounder but no effect
    between  the treatment and outcome variables.

    Where there are many such pairs, we sample one pair randomly.

    """
    all_nodes = list(graph.nodes)
    all_edges = list(graph.edges)

    variables_of_interest = [(a_[0],a_[1],node) if a_ not in all_edges else None for node in all_nodes for a_ in get_pairwise_elements(get_adj_of_edge(node,graph)) ]
    variables_of_interest = [v for v in variables_of_interest if v is not None]

    return variables_of_interest

def treatment_effect_with_mediator(graph):

    """
    Given a graph, we want to find one treatment and effect satisfying the criterion of having a mediator variable.

    Where there are many such pairs, we sample one pair randomly.

    """
    all_edges = list(graph.edges)
    edges_with_mediator = [(edge[0],edge[1],list(nx.all_simple_paths(graph,edge[0],edge[1]))) for edge in all_edges ]
    variables_of_interest = [(e[0],e[1],e[2]) for e in edges_with_mediator if e[2]!=[] and len(e[2])>1]
    return variables_of_interest

def treatment_effect_with_common_child(graph):

    """
    Given a graph, we want to find one treatment and effect satisfying the criterion of 
    the the treatment and outcome variables having the same child variable.

    Where there are many such pairs, we sample one pair randomly.

    """
    all_edges = list(graph.edges)
    confounders = [(edge[0],edge[1],set(get_adj_of_edge(edge[0],graph)).intersection(set(get_adj_of_edge(edge[1],graph)))) for edge in all_edges]
    variables_of_interest = [c for c in confounders if len(c[2])!=0]
    return variables_of_interest

@wrap_non_picklable_objects
def get_true_edge_coefficients(graph):
    # Given the true graph that has been unpickled, we want to get all its edge coefficients
    with open('graph.pkl','rb') as f:
        graph = pl.load(f)
@wrap_non_picklable_objects
def convert_posterior_to_graph(posterior,index_to_node):
    graph_sample = nx.from_numpy_array(posterior,create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(graph_sample): # Check it is acyclic DAG
        return None
    graph_sample_relabeled = nx.relabel_nodes(graph_sample, index_to_node)
    return graph_sample_relabeled
   

@wrap_non_picklable_objects
def get_causal_estimate(graph,df,VARIABLE_DICT):
    model = CausalModel(data=df, treatment=[VARIABLE_DICT['source']],outcome=VARIABLE_DICT['target'],graph=nx.DiGraph(graph),use_graph_as_is=True)
    # II. Identify causal effect and return target estimands
    identified_estimand = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(identified_estimand,
            target_units='ate',
            control_value=0.0,
            treatment_value=1.0,
            method_name="backdoor.linear_regression",
            test_significance=False,confidence_intervals=False)
    causal_estimate = causal_estimate_reg.value
    return causal_estimate

@wrap_non_picklable_objects
def get_estimate_from_posterior(each_posterior,index_to_node,BASE_PATH,VARIABLE_DICT):
    df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))

    graph_sample = nx.from_numpy_array(each_posterior,create_using=nx.DiGraph)
    if nx.is_directed_acyclic_graph(graph_sample): # Check it is acyclic DAG
        graph_sample_relabeled = nx.relabel_nodes(graph_sample, index_to_node)
        estimate = get_causal_estimate(graph_sample_relabeled,df,VARIABLE_DICT)
        return estimate
    else:
        return None    

@wrap_non_picklable_objects
def get_true_causal_estimate(true_graph,BASE_PATH,VARIABLE_DICT):
    
    source = VARIABLE_DICT['source']
    target = VARIABLE_DICT['target']
    # The true causal coefficients are given, so we extract them, and do not need to do linear regression
    # go to the path and extract the json file. if the json does not exsit,...
    # then use path tracing, giving the source and target variables to get the correct estimate
    true_weights_json_path = os.path.join(BASE_PATH,'true_edge_weights.json')
    if not os.path.exists(true_weights_json_path):
        raise Exception(f'The JSON file for the weights of the true causal graph does not exist. Can not proceed further.')
    
    with open(true_weights_json_path,'r') as f:
        graph_weights = json.load(f)

    # Transform weights to a dictionary, to make search easier
    graph_weights_dict = {f'{w[0].strip()}-{w[1].strip()}':w[2] for w in graph_weights}
    
    total_weight = 0
    all_paths = list(nx.all_simple_paths(true_graph,source,target))
    for path in all_paths:
        w=1
        for i in range(len(path)-1):
            variable_string = f'{path[i].strip()}-{path[i+1].strip()}'
            try:
                w = w * graph_weights_dict[variable_string]
            except KeyError:
                # The given source -> target does not exist in the true causal graph
                w = 0
        total_weight+=w
    return total_weight


def calculate_rmse(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the root mean squared error (RMSE) between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    # Remove `None` values from the RMSE calculation: encountered it in `dibs`
    mask = a!=None
    a_ = a[mask]
    b_ = b[mask]
    ####################
    return np.sqrt(np.mean(np.square(np.subtract(a_, b_)), axis=axis))


def calculate_squared_diff(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the squared difference between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.square(np.subtract(a, b))


if __name__=="__main__":
    baseline_to_use = sys.argv[1]
    CAUSAL_INFERENCE_CASE = sys.argv[3]
    VARIABLE_SELECTION_MAPPING = {
    'treatment_effect_with_common_child':treatment_effect_with_common_child,
    'treatment_effect_with_mediator':treatment_effect_with_mediator,
    'treatment_effect_with_cofounder_no_effect':treatment_effect_with_cofounder_no_effect,
    'treatment_effect_with_cofounder':treatment_effect_with_cofounder
    }
    SEED_TO_USE = []
    CAUSAL_INFERENCE_CASE_FUNCTION = VARIABLE_SELECTION_MAPPING[CAUSAL_INFERENCE_CASE]

    seed_number = int(sys.argv[2])
    if seed_number != 0:
        SEED_TO_USE = [i for i in range(seed_number-5,seed_number)]

 

    for seed in SEED_TO_USE:
        rng = default_rng(seed)

        # Use a true graph from bcdnets just to get the variables of interest
        BASE_PATH_ = os.path.join(os.path.join(BASELINE_FOLDER,'bcdnets'),str(seed))

        with open(os.path.join(BASE_PATH_,'graph.pkl'),'rb') as fl:
            true_graph_ = pl.load(fl)

        variables_of_interest_list = CAUSAL_INFERENCE_CASE_FUNCTION(true_graph_)
        random_choice = rng.integers(low=0,high = len(variables_of_interest_list)) #persistent across seeds

        variables_of_interest = variables_of_interest_list[random_choice]
        SOURCE_VARIABLE = variables_of_interest[0]
        TARGET_VARIABLE = variables_of_interest[1]

        VARIABLE_DICT = {
            'source':SOURCE_VARIABLE,
            'target':TARGET_VARIABLE
        }

        
        for baseline in [baseline_to_use]:
            BASE_PATH = os.path.join(os.path.join(BASELINE_FOLDER,baseline),str(seed))

            if not os.path.exists(BASE_PATH):
                continue

            with open(os.path.join(BASE_PATH,'graph.pkl'),'rb') as fl:
                true_graph = pl.load(fl)

            print('='*100)
            print(f'Using case: {CAUSAL_INFERENCE_CASE}\nSeed: {seed}\nBaseline: {baseline} \nInteresting variables found {VARIABLE_DICT}')
            print('='*100)

            # Get posterior
            count=0
            posterior_file_path = os.path.join(BASE_PATH,'posterior.npy')
            if not os.path.isfile(posterior_file_path):
                continue
            posterior = np.load(posterior_file_path)
            #posterior = np.load(posterior_file_path)[:1,:,:] # for debugging
            
            # Without parallelization
            #causal_estimates = np.array([get_estimate_from_posterior(posterior[i,:,:],index_to_node,BASE_PATH) for i in range(posterior.shape[0])])

            results = Parallel(n_jobs=len(os.sched_getaffinity(0)))(
                    delayed(get_estimate_from_posterior)(posterior[i,:,:],index_to_node,BASE_PATH,VARIABLE_DICT)
                    for i in range(posterior.shape[0])
                    )
            causal_estimates = np.asarray(results)
            df = pd.read_csv(os.path.join(BASE_PATH,'data.csv'))

            true_estimate = get_true_causal_estimate(true_graph,BASE_PATH,VARIABLE_DICT)

            with open(f'/home/mila/c/chris.emezue/scratch/causal_inference/{OUTPUT_FOLDER_NAME}/{baseline_to_use}_{seed}_{CAUSAL_INFERENCE_CASE}_ate_estimates.npy', 'wb') as fl:
                np.save(fl,causal_estimates)
            true_causal_estimates = np.full(causal_estimates.shape, fill_value=true_estimate)
            with open(f'/home/mila/c/chris.emezue/scratch/causal_inference/{OUTPUT_FOLDER_NAME}/true_{baseline_to_use}_{seed}_{CAUSAL_INFERENCE_CASE}_ate_estimates.npy', 'wb') as fl:
                np.save(fl,true_causal_estimates)
            rmse_value = calculate_rmse(causal_estimates,true_causal_estimates)
            
            baselines_.append(baseline)
            rmses_.append(rmse_value)
            seeds_.append(seed)

    if len(baselines_)!=0 and len(rmses_)!=0 and len(seeds_)!=0:  
        assert len(baselines_)==len(rmses_) == len(seeds_)
        case_list = [CAUSAL_INFERENCE_CASE for i in range(len(rmses_))]
      
        df = pd.DataFrame({'baselines':baselines_,'rmse':rmses_,'seeds':seeds_,'cases':case_list})
        df.to_csv(f'{OUTPUT_FOLDER_NAME}/{baseline_to_use}_{SEED_TO_USE}_{CAUSAL_INFERENCE_CASE}_ate_estimates.csv',index=False)
    else:
        print("List was empty so nothing was done")
    print(f'ALL DONE for seeds: {SEED_TO_USE}') 