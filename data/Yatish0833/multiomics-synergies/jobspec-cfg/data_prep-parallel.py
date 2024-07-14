#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You may need to remove the '#' from the following commands to install the required dependencies

#these two are to read the excel
#! pip install xlrd
#! pip install install openpyxl


# In[ ]:


## ***** THESE ARE THE DEFAULTS UNLESS THEY ARE CHANGED WHEN YOU RUN THE CODE!!! *****

#Best overall parameters: min_repetitions = 2, n_trees = 6000, mtry = 500, max_depth = 10, min_node= 15

min_repetitions = 2 #Number of repetitions an interaction appears in the trees
max_order = 4 
working_dir = 'rocstool/tmp/' #make sure it is empty
drop_nans = False

#Ranger parameters
n_trees = 1000 #Number of trees
mtry = 100 # Number of variables to possibly split at in each node. Default is the (rounded down) square root of the number variables. 
max_depth = 0 # Maximal tree depth. A value of NULL or 0 (the default) corresponds to unlimited depth, 1 to tree stumps (1 split per tree).
min_node= 5 # Minimal node size. default ranger: 5

# File inputs
x_input = 'data/train_protein_matrix.csv'
y_input = '/datasets/work/hb-procan/work/data/DrugResponse_PANCANCER_GDSC1_GDSC2_20200602.csv'

test_input = 'data/test_protein_matrix.csv'


# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import os
import sys
import argparse
import shutil

import statsmodels.formula.api as smf
from itertools import combinations


# In[ ]:


# Defining the number of splits in the data
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=1)
    parser.add_argument('--nSplits', default=1) #This makes it not parallel
    parser.add_argument('--minRep', default=min_repetitions)
    parser.add_argument('--maxOrder', default=max_order)
    parser.add_argument('--nTrees', default=n_trees)
    parser.add_argument('--mtry', default=mtry)
    parser.add_argument('--maxDepth', default=max_depth)
    parser.add_argument('--minNode', default=min_node)
    parser.add_argument('--workingDir', default=working_dir)
    
    
    parser_args = parser.parse_args()
    
    
    split_nr = int(parser_args.split)
    n_splits = int(parser_args.nSplits)
    
    
    min_repetitions = int(parser_args.minRep) 
    max_order = int(parser_args.maxOrder)
    working_dir = parser_args.workingDir

    #Ranger parameters
    n_trees = int(parser_args.nTrees)
    mtry = int(parser_args.mtry)
    max_depth = int(parser_args.maxDepth)
    min_node= int(parser_args.minNode)


except Exception as e: 
    print(e)
    split_nr = 1
    n_splits = 2
    print('Did you forget to add the split number and the number of splits?')
    #exit()

    


# In[ ]:


# If this split finalized successfully, it does not re-run
if os.path.exists(working_dir+f"final_results{split_nr}.tsv") and os.path.exists(working_dir+f"tree_performances{split_nr}.tsv"):
    print('Job previously run successfully!\nExiting')
    exit()

if os.path.exists(working_dir+f"final_results.tsv") and os.path.exists(working_dir+f"tree_performances.tsv"):
    print('Job previously run successfully!\nExiting')
    exit()


# In[ ]:


# Easing things for other systems
dir_trees_tmp = working_dir+"tmp" # temportal directory where each of the separate trees are going to be saved
#final_results_dir ="tmp/" # directory where the final results are going to be saved
path_to_R = 'Rscript' # Path to run R
path_to_ranger_script = 'ranger_run-parallel.R' # Path to the ranger script


# In[ ]:


#One row per cell line
#DIR
# x = pd.read_excel(x_input, engine='openpyxl').drop(columns=['Project_Identifier'])
# c = [a.replace(';','.') for a in x.columns]
# x.columns = c
# x.columns
x = pd.read_csv(x_input)

test = pd.read_csv(test_input)
# In[ ]:


#DIR
y = pd.read_csv(y_input)[['drug_id','cell_line_name','ln_IC50', 'max_screening_conc']]
y.columns


# In[ ]:


# This cell is the function to go from the table to a JSON file (variantSpark format and structure)

def merge_tree_node(tree, node):
    
    # Empty tree
    if type(tree)==float: return tree
    if len(tree)==0: return node

    # Direct children
    if tree['right'] == node['nodeID']:
        tree['right'] = node
        return tree
    elif tree['left'] == node['nodeID']:
        tree['left'] = node
        return tree

    # Create
    right = merge_tree_node(tree['right'], node)
    left = merge_tree_node(tree['left'], node)
    tree['right'] = right
    tree['left'] = left
    return tree
            

def from_table_to_json(m):
    tree = {}
    for _id,row in m.iterrows():
        current_node = {'nodeID': row['nodeID'], 
                        'splitvarID':row['splitvarID'],
                        'splitVar':row['splitvarName'],
                        'splitval':row['splitval'], 
                        'terminal':row['terminal'], 
                        'prediction':row['prediction'], 
                        'left':row['leftChild'], 
                        'right':row['rightChild'] }
        tree = merge_tree_node(tree, current_node)
    return tree



# Test
#m = pd.read_csv('output/tree1.csv')
#from_table_to_json(m)


# In[ ]:


# Algorithm to get the interacting nodes (no testing done yet)

def get_interactions(tree, current_list, interactions):
    if not 'splitVar' in tree.keys():
        return 0
    if str(tree['splitVar']) == 'nan': return 0 #ranger adds a fake predicting node at the end
    
    # Adding the interaction
    current_list.append(tree['splitVar'])
    if len(current_list) >= 2:
        for i in range(2,len(current_list)+1):
              if len(current_list[-i-1:]) == len(set(current_list[-i-1:])) and "maxscreeningconc" not in current_list:
                aux = '+'.join(sorted(current_list[-i:]))
                if aux in interactions.keys():
                    interactions[aux] +=1
                else:
                    interactions[aux] = 1
                    
    if 'left' in tree.keys():
        get_interactions(tree['left'], current_list, interactions)
    if 'right' in tree.keys():
        get_interactions(tree['right'], current_list, interactions)
        
    _ = current_list.pop()
    


# In[ ]:


def undo(string):
    
    string = ''.join([ x if ord(x)<90 else str(ord(x)-97) for x in string ])
    string = string[:6]+'.'+string[6:].replace('HUMAN', '_HUMAN') #not sure these 6
    return string
    
#undo('PacfbbCRYABHUMAN')


# In[ ]:


# Testing all the interactions

def results_fit_to_df(results):
    coeffs = results.params.tolist()
    pvals = results.pvalues.tolist()
    pseudo_r2 = results.rsquared
    tvals = results.tvalues.tolist()
    cint_low = results.conf_int()[0].tolist()
    cint_high = results.conf_int()[1].tolist()

    try:
        results = results.summary()
    except:
        #ValueError: resids must contain at least 2 elements
        r = pd.DataFrame([1,2,3]) #dirty...
        r['z']='nan'
        return r
    converged = results.tables[0].data[5][1].strip()
    results = results.tables[1].data
    results = pd.DataFrame(results[1:], columns=['coef_id', 'coef', 'std err', 'z', 'P>|z|', '[0.025', '0.975]'])
    results['P>|z|'] = pvals
    results['z'] = tvals 
    results['coef'] = coeffs
    results['converged'] = converged
    results['pseudo_r2'] = pseudo_r2
    results['[0.025'] = cint_low
    results['0.975]'] = cint_high
    return results
    
def test_interactions_high(df, data, max_order=4, repetitions_threshold=2, min_samples=20):
    """
    I use GLM because:
    The main difference between the two approaches is that the general linear model strictly assumes that
    the residuals will follow a conditionally normal distribution, while the GLM loosens this assumption 
    and allows for a variety of other distributions from the exponential family for the residuals.
    """
    final_results = []
    counts = 0

    for m_or in range(2,max_order+1):
        #print('current order',m_or)
        
        for v in df[(df.repetitions>=repetitions_threshold) & (df.order==m_or)].variants.tolist():
            #preparing the input
            sp=v.split('+')
            if len(sp) != len(set(sp)): continue
            xy = data[sp+["max_screening_conc", 'ln_IC50']]
            if drop_nans:
                xy = xy.dropna()
            if len(xy.columns) <= m_or: continue #not enough columns # how would we get that case? deccision over the same protein several times?
            if len(xy) <= min_samples: continue #not enough rows
            sp=v.replace('_','').split('+')
            xy.columns = [''.join([chr(int(y)+97) if y.isnumeric() else y for y in x.replace('_','').replace('.','')]) for x in xy.columns]
            formula = xy.columns[-1]+' ~ '
            for i in range(1,len(xy.columns)):
                formula = formula + ' + '.join(['*'.join(o) for o in list(combinations(xy.columns[:-2],i))])
                formula = formula + ' + '
            formula = formula.rstrip(' + ')
            
            #Recreating the formula
            if m_or>2:
                #gathering all interactions
                fs = formula.split(' + ')
                formula = ' + '.join([a for a in fs if '*' not in a]+[a for a in fs if a.count('*')== m_or-1])
                all_interactions = [a.replace('*',':') for a in fs if '*' in a]
                final_results = pd.concat(final_results)
                subset = final_results[final_results.coef_id.apply(lambda a: a in all_interactions)].reset_index(drop=True)

                final_results = [final_results]
                if len(subset)>0:
                    max_idx = subset['coef'].astype(float).abs().idxmax()
                    coef_id = subset.loc[max_idx].coef_id
                    formula = formula +' + '+coef_id.replace(':','*')
                else:
                    #pass
                    continue # bc i dont think it is a valid tree form (interaction-wise)
                    #There is no sub epistasis (P>Q>O>P, tree 503, first compound)

            # Standard fitting
            try:
                ols = smf.ols(formula.replace('*',':') + " + maxscreeningconc",data=xy)
                # "*" vs ":" #https://stackoverflow.com/questions/33050104/difference-between-the-interaction-and-term-for-formulas-in-statsmodels-ols
            except:
                print('error in OLS')
                #print('coef_id',coef_id)
                print('formula OLS',type(formula),formula)
                #return pd.concat(final_results)
                continue
            ols.raise_on_perfect_prediction = False #preventing the perfect separation error
            results = ols.fit(disp=False, maxiter=1000) #mehtod prevents singular matrix
#            print(formula)
#            return results
            results = results_fit_to_df(results)
            results['standard_fitting'] = True

            #If nan means no convergence bc singular matrix
            #adding regularization
            if 'nan' == pd.DataFrame(results)['z'].astype(str).iloc[2].strip():
                try:
                    results = ols.fit_regularized(method='l1', disp=False, maxiter=1000, alpha=0.3) #mehtod prevents singular matrix
                    results = results_fit_to_df(results)
                    results['standard_fitting'] = False        
                except:
                    #crashed the regularized
                    counts +=1
                    continue


            results['snps'] = v
            results['order'] = len(sp)
            final_results.append(results)
    try:
        final_results = pd.concat(final_results)  # this works for any amount of results, but none
    except:
        print('no results for this drug')
        return pd.DataFrame(final_results)  # returns empty data frame
    return final_results


# In[ ]:


#TODO: By compound

#Looping over all drugs
# drug_id
# Other options:
# - drug_name
# - CHEMBL = Chemical compound ID
#for compound_name, group in x.merge(y, left_on='Cell_Line', right_on='cell_line_name').groupby('drug_id'): # may require too much memory

#Making a temp file to run all R stuff
#DIR
if not os.path.exists(dir_trees_tmp+f"{split_nr}"):
    os.makedirs(dir_trees_tmp+f"{split_nr}")


column_to_group = 'drug_id'
drugs_list = y[column_to_group].sort_values().unique()
drugs_list = [drugs_list[i] for i in range(len(drugs_list)) if i%n_splits==split_nr-1]
i = -1
all_drug_results = []
tree_performances = []
for elm in drugs_list:
    print(f"{path_to_R} {path_to_ranger_script}  -w {working_dir} -c {split_nr} -n {n_trees} -t {mtry} -s {min_node} -d {max_depth}")
    i+=1
    
    if i%10==0 or i<10: print(i,'out of',len(drugs_list), 'drugs in split nr', split_nr)

    xy = x.merge(y[y[column_to_group]==elm], left_on='Cell_Line', right_on='cell_line_name')
    #Enhancement: Remove peptides that are all zero 
    test_xy = test.merge(y[y[column_to_group]==elm], left_on='Cell_Line', right_on='cell_line_name')
    # saving csv for R df
    # file name is generic but we could personalize it
    #DIR
    xy.drop(columns=['Cell_Line', 'cell_line_name','drug_id']).fillna(0).rename(columns={'ln_IC50':'label'}).to_csv(dir_trees_tmp+f"{split_nr}/data.csv", index=False)

    test_xy.drop(columns=['Cell_Line', 'cell_line_name','drug_id']).fillna(0).rename(columns={'ln_IC50':'label'}).to_csv(dir_trees_tmp+f"{split_nr}/test_data.csv", index=False)
    #Run the R script to generate the outputs
    #os.system(f"{path_to_R} {path_to_ranger_script}  -w {working_dir} -c {split_nr} -n {n_trees} -t {mtry} -s {min_node} -d {max_depth}")
    print("R output (in case there is an error or something)")
    #os.popen(f"{path_to_R} {path_to_ranger_script}  -w {working_dir} -c {split_nr} -n {n_trees} -t {mtry} -s {min_node} -d {max_depth}").read()    
    print(os.popen(f"{path_to_R} {path_to_ranger_script}  -w {working_dir} -c {split_nr} -n {n_trees} -t {mtry} -s {min_node} -d {max_depth}").read())
    
    #load the R outputs (the trees, one file each), and convert it to VS look-alike and get interactions
    interactions = {}
  
    aggregated_trees_df = pd.read_csv(os.path.join(dir_trees_tmp+str(split_nr),'aggregated_trees.csv'))
    aggregated_trees_list = aggregated_trees_df.tree.unique()
    for tree_nr in aggregated_trees_list:
        tree_df = aggregated_trees_df[aggregated_trees_df.tree==tree_nr]
        tree_json = from_table_to_json(tree_df)        
        get_interactions(tree_json,[],interactions)
    
    # Creating a df out of the interactions
    df = pd.DataFrame({'variants':interactions.keys(),'repetitions':interactions.values()})
    df['order'] = df.variants.apply(lambda x: x.count('+')+1)
    
    #get tree performances
    aux_performances = pd.read_csv(os.path.join(dir_trees_tmp+str(split_nr),"performance.tsv"), sep='\t')
    aux_performances['drug'] = elm
    if os.path.isfile(working_dir+f"tree_performances{split_nr}.tsv"):
        aux_performances.to_csv(working_dir+f"tree_performances{split_nr}.tsv", index=False, sep='\t',mode='a', header=False)
    else:
        aux_performances.to_csv(working_dir+f"tree_performances{split_nr}.tsv", index=False, sep='\t',mode='a')
    
    tested_interactions = test_interactions_high(df, xy, max_order=max_order, repetitions_threshold=min_repetitions) #here you define which order of interactions you want to compute
    tested_interactions['drug'] = elm
    if os.path.isfile(working_dir+f"final_results{split_nr}.tsv"):
        tested_interactions.to_csv(working_dir+f"final_results{split_nr}.tsv", index=False, sep='\t',mode='a', header=False)
    else:
        tested_interactions.to_csv(working_dir+f"final_results{split_nr}.tsv", index=False, sep='\t',mode='a')
#    if i==2: break
    


# deliting temp folder from raneger
shutil.rmtree(dir_trees_tmp+f"{split_nr}")
#Option B: os.system("rm -rf _path_to_dir")


# In[ ]:


print('Split ',split_nr, 'out of ',n_splits,' is DONE')


# In[ ]:


#All jobs finished and produced outputs
for i in range(1,n_splits+1):
    if os.path.exists(dir_trees_tmp+f"{i}"): exit()
    if not os.path.exists(working_dir+f"final_results{i}.tsv"): exit()
    

fr = [x for x in os.listdir(working_dir) if 'final_results' in x and 'final_results_all.tsv' not in x] #all jobs created the file
if len(fr) == n_splits:

    #appending everything together and Removing temp final results
    for final_result in fr:
        if os.path.exists(working_dir+f"final_results_all.tsv"):
            pd.read_csv(os.path.join(working_dir,final_result), sep='\t').to_csv(working_dir+f"final_results_all.tsv", index=False, sep='\t',mode='a', header=False)
        else:
            pd.read_csv(os.path.join(working_dir,final_result), sep='\t').to_csv(working_dir+f"final_results_all.tsv", index=False, sep='\t',mode='a')
        os.remove(os.path.join(working_dir,final_result))
        #pass
        
    # now the same for performances    
    #Adding tree performances and Removing temp final results
    pr = [x for x in os.listdir(working_dir) if 'tree_performances' in x and 'tree_performances_all.tsv' not in x]
    for tree_performance in pr:
        if os.path.exists(working_dir+f"tree_performances_all.tsv"):
            pd.read_csv(os.path.join(working_dir,tree_performance), sep='\t').to_csv(working_dir+f"tree_performances_all.tsv", index=False, sep='\t',mode='a', header=False)
        else:
            pd.read_csv(os.path.join(working_dir,tree_performance), sep='\t').to_csv(working_dir+f"tree_performances_all.tsv", index=False, sep='\t',mode='a')
        #pd.read_csv(os.path.join(working_dir,tree_performance), sep='\t').to_csv(working_dir+f"tree_performances_all.tsv", index=False, sep='\t',mode='a')
        os.remove(os.path.join(working_dir,tree_performance))
        #pass

    print('All jobs finished successfully!\n final_results_all.tsv has all the aggregated output')

