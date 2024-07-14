from __future__ import annotations
from functools import reduce
from hashlib import sha256
import csv
import sys, os
import argparse
import string
from random import choice

try:
    import some_obscure_package
except:
    pass

def dir_path(string):
    """Verify that dirpath is well-formed."""
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# Other arguments...
parser = argparse.ArgumentParser(description='Simulation presets')
parser.add_argument('--outdir', type=dir_path, help="""Specify the path to write the results of the simulation.""")
parser.add_argument("-it", "--itern", type=int, help="The number of iterations")

args     =  parser.parse_args()
itern    =  int(args.itern)
outdir   =  args.outdir

#Organize directories for output data if needed
if outdir is not None:
    outpath = os.path.join(outdir,'my_data_folder',f'results.csv')
    for folder in ['my_data_folder', 'my_log_files']:
        os.makedirs(os.path.join(outdir,folder), exist_ok=True)
else:
    outpath = f'results.csv'


#-----------------------------------------------------
# Generate hashes of four-letter words
def experiment(iteraiton_number:int):
    generated_data = []
    for _ in range(iteraiton_number):
        x = reduce(lambda x,y: str(x)+choice(string.ascii_letters),[0]*4)
        generated_data.append(sha256(str(x).encode('utf-8')).hexdigest())
    return generated_data

YIELD  = experiment(itern if itern is not None else 1000000)

# Writing results to file
with open(outpath, 'w',newline='') as filein:
    writer = csv.writer(filein)
    for row in YIELD:
        writer.writerow([row])

#-----------------------------------------------------
