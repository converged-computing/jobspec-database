import glob
import re
import time
import os
from joblib import Parallel, delayed
import pickle
import numpy as np
import pandas as pd
import glob
import sys
import argparse
parser = argparse.ArgumentParser()

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

start = int(sys.argv[1])
end = int(sys.argv[2])
run = 'runBC'
labels_df = pd.read_csv(f'/hercules/scratch/atya/BinaryML/meta_data/labels_{run}.csv')
labels_df = labels_df[labels_df['z_max_rel_from_pvol'].isna()]
# start = 0
# end = 400
cur_dir = '/hercules/results/atya/BinaryML/'

myexecute(f'mkdir -p {cur_dir}sims/{run}/pdot_vol')
files_to_process = labels_df['file_name']#.values[start:end]
periods = labels_df['p_middle']#.values[start:end]

myexecute('echo \"\n\n\n\n############################################################################## \n\n\n \
          Comment: Estimating the pdot vol of all simulations:\
           \n\n\n ############################################################################## \n\n\n \"')

def prepfold(period,file_to_process):
    sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /u/pdeni/fold-tools-2020-11-18-4cca94447feb.simg '
    #sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /hercules/scratch/atya/compare_pulsar_search_algorithms.simg '
    file_loc = cur_dir+f'sims/{run}/dat_inf_files/'+file_to_process
    out = cur_dir+f'sims/{run}/pdot_vol/'+file_to_process[:-4]
    #print(out)
    myexecute(sing_prefix+f'python gen_pdot_vol.py {period} {file_loc} {out}')

# dat_files = glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.dat')
# dat_files.extend(glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.inf'))
# #dat_files = ['/hercules/scratch/atya/BinaryML/sims/obs2C.dat','/hercules/scratch/atya/BinaryML/sims/obs2C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.dat']

val = Parallel(n_jobs=-1)(delayed(prepfold)(period = period,file_to_process=file_to_process) for period,file_to_process in zip(periods,files_to_process))

