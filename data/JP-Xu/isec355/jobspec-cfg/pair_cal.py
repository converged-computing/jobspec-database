import numpy as np
import pandas as pd
import MDAnalysis as mda
import os
#import csv
import matplotlib.pyplot as plt
#from multiprocessing import Pool, TimeoutError
#import seaborn as sns
import natsort
import regex as re
#from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import time

def update_pdb_Bfactor(filename):
    """
    This function updates B factor column of its input pdb file with following rules:
    Add headgroup, phosphate group, glycerol group, and tail group for lipid with PS, PA, PC, PE, PG, TAP in its residue name.
    Add individual groups for other residues.
    
    Input filename: file name of raw pdb file
    Return: a dictionary for B factor index and its corresponding group
    
    Note: New pdb file will be generated directly under current folder, with _B_factor addition before ".pdb".
    
    """
    
    u = mda.Universe(filename)
    selection_strings = {}

    ### ----- Generate selection strings  ----- ###

    for rs in pd.unique(u.atoms.resnames):
        # regex checks if it's a lipid
        if re.search('(PC|PA|PE|PS|PG|TA)', rs):
            lipid_atom_names = pd.unique(u.select_atoms('resname '+rs).names)
            try: 
                P_index = np.where(lipid_atom_names == 'P')[0][0] 
            except: 
                P_index = 0

            try:
                C1_index = np.where(lipid_atom_names == 'C1')[0][0]
            except:
                C1_index = 0

            try:
                C2_index = np.where(lipid_atom_names == 'C2')[0][0]
            except:
                C2_index = 0
            
            C32_index = np.where(lipid_atom_names == 'C32')[0][0]

            if P_index != 0:  # If there is a phosphate group, like PA, PC, PE, PG and PS.
                selection_strings[rs+'_head_string'] = 'resname ' + rs + ' and (name ' + ' or name '.join(lipid_atom_names[:P_index]) + ')'
                selection_strings[rs+'_phos_string'] = 'resname ' + rs + ' and (name ' + ' or name '.join(lipid_atom_names[P_index:C1_index]) + ')'
                selection_strings[rs+'_glyc_string'] = 'resname ' + rs + ' and (name ' + ' or name '.join(lipid_atom_names[C1_index:C32_index]) + ')'
            
            elif C2_index != 0:
                selection_strings[rs+'_head_string'] = 'resname ' + rs + ' and (name ' + ' or name '.join(lipid_atom_names[:C2_index]) + ')'
                selection_strings[rs+'_glyc_string'] = 'resname ' + rs + ' and (name ' + ' or name '.join(lipid_atom_names[C2_index:C32_index]) + ')'
            
            
            selection_strings[rs+'_tail_string'] = 'resname ' + rs + ' and (name ' + ' or name '.join(lipid_atom_names[C32_index:]) + ')'
        else:
            selection_strings[rs] = 'resname ' + rs

    ### ----- Update B group  ----- ###

    ft = 1
    bf_group = {}
    
    for k, v in selection_strings.items():
        u.select_atoms(v).tempfactors = ft
        
        if '_' in k:
            bf_group["_".join(k.split("_")[:-1])] = ft
        else:
            bf_group[k] = ft
        ft += 1

    ### ----- Save updated pdb file ----- ###
    insert_string = "_B_factor."
    output_string = re.sub(r'\.(?=[^\.]*$)', insert_string, filename)
    print("Processed pdb file has been saved to "+output_string)
    u.atoms.write(output_string)
    
    return bf_group


def get_files(pre = 'gamma', ext = 'dcd'):
    ''' This function returns files under current folder with defined prefix and extension.
    input pre: string, prefix of wanted files
    input ext: string, extension of wanted files.
    return: a list of file names.
    '''
    result = []
    for file_ in os.scandir():
        if file_.is_file() and re.search(r'^(%s)'%pre, file_.name) and re.search(r'(%s)$'%ext, file_.name):
            result.append(file_.name)
    result = natsort.natsorted(result, alg=natsort.FLOAT)        
    return result

def get_dirs(pre = '65', ext = 'input'):
    ''' This function returns files under current folder with defined prefix and extension.
    input pre: string, prefix of wanted files
    input ext: string, extension of wanted files.
    return: a list of file names.
    '''
    result = []
    for file_ in os.scandir():
        if file_.is_dir() and file_.name.startswith(pre) and file_.name.endswith(ext):
            result.append(file_.name)
    result = natsort.natsorted(result, alg=natsort.FLOAT)        
    return result

if __name__ == "__main__":
    surf_tensions = [-7, 0, 7, 15]
    os.system('cp -rf $HOME/toppar ./')
    for st in surf_tensions:
        os.system(f"mdconvert gamma{st}*.dcd -c 100000 -o gamma{st}_total.dcd")
        input_file = f'gamma{st}.inp'
        # Add rerun commands
#        with open(input_file, 'a') as f:
#            print(rerun_tcl_string%(st), file=f)
        # update pdb file's B factor
        b2g = update_pdb_Bfactor('step5_input.pdb')
        group_names = list(b2g.keys())
        group_ndces = list(b2g.values())
        xscfiles = get_files(pre=f'gamma{st}\.part00[0-9]', ext='restart.xsc')
        os.system(f"sed -i '/set inputname /s/.*/set inputname  {xscfiles[-1][:-4]};/g' gamma{st}.inp")
        for i, group1_name in enumerate(group_names):
            #for j, group2_name in enumerate(group_names[i:]):
            time.sleep(1)
            output_file = "gamma{0}_{1}_{2}.log".format(st, group1_name, group1_name)
            # Change group index
            time.sleep(1)
            os.system("sed -i '/pairInteractionGroup1 /s/.*/pairInteractionGroup1  {}/g' {}".format(group_ndces[i], input_file))
            time.sleep(1)
            #os.system("sed -i '/pairInteractionGroup2 /s/.*/pairInteractionGroup2  {}/g' {}".format(group_ndces[i+j], input_file))
            #time.sleep(1)
            os.system("charmrun +p28 namd2 {0} > {1}".format(input_file, output_file))
            os.system(f"grep ENERGY -h {output_file} > clean_{output_file}")


