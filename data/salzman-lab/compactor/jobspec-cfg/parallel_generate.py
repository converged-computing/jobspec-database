import numpy as np
import pandas as pd
import glob
import os
import subprocess
import sys
import time
import argparse

anchors_to_process_in_parallel = 80

# Function to submit parallel slurm jobs which generate compactors from intermediate files. 
def sbatch_file(intermediate, file_name):

    job_file = open(file_name + '.sh', "w")
    job_file.write("#!/bin/bash\n#\n")
    job_file.write("#SBATCH --job-name=" + sys.argv[3] + "\n")
    job_file.write("#SBATCH --time=1:00:00" + "\n")
    job_file.write("#SBATCH -p horence,normal,quake\n")
    job_file.write("#SBATCH --nodes=1\n")
    job_file.write("#SBATCH --mem=16000\n") 
    job_file.write('ml python/3.6.1\nml py-numpy\nml py-pandas\nml biology\nml py-biopython/1.79_py39\n')
    job_file.write("python3 aspen.py " + intermediate + " " + sys.argv[1] + " " + sys.argv[2])
    job_file.close()
    status, job_num = subprocess.getstatusoutput("sbatch {}".format(file_name + '.sh'))
    
intermediates = [i for i in os.listdir() if '.intermediary' in i]

count_dictionary = {}

header = 'anchor' + '\t' + 'compactor_valid' + '\t' + 'majority_local_proportion' + '\t' + 'anchor_abundance' + '\t' + 'valid_local_proportion'  + '\t' + 'valid_abundance' + '\t' + 'majority_abundance' + '\t' + 'effect_size' + '\t' + 'mean_target_levenshtein_distance' + '\t' +  'path' + '\t' + 'compactor_majority' + '\n'

if True:
    with open('compactor_summary.tsv', 'a') as consensus_log:
        consensus_log.write(header)
    consensus_log.close()


for i in range(0, len(intermediates), anchors_to_process_in_parallel):
    sbatch_file(",".join(intermediates[i:i+anchors_to_process_in_parallel]), 'generate_compactor_' + str(i))
