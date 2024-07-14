import numpy as np
import pandas as pd
import glob
import os
import subprocess
import sys
import time
import argparse

fastqs_to_process_in_parallel = 1

# Function to write and submit slurm jobs which parse samplesheet fastqs in parallel.
def sbatch_file(fastq, file_name, num_fastqs):

    job_file = open(file_name + '.sh', "w")
    job_file.write("#!/bin/bash\n#\n")
    job_file.write("#SBATCH --job-name=" + sys.argv[4] + "\n")
    job_file.write("#SBATCH --time=01:00:00" + "\n")
    job_file.write("#SBATCH -p horence,owners,quake\n")
    job_file.write("#SBATCH -p horence,owners\n")
    job_file.write("#SBATCH --nodes=1\n")
    job_file.write("#SBATCH --mem=16000\n") 
    job_file.write('ml python/3.6.1\nml py-numpy\nml py-pandas\nml biology\nml py-biopython/1.79_py39\n')
    job_file.write("python3 make_intermediates.py " + fastq + " " + str(num_fastqs) + " " + sys.argv[3])
    job_file.close()
    status, job_num = subprocess.getstatusoutput("sbatch {}".format(file_name + '.sh'))

    
    
# Load samplesheet. 
samplesheet = pd.read_csv(sys.argv[1], engine='python', sep=',', header=None)
fastqs = list(samplesheet.iloc[:,0])

# Write sample contribution file. 
contribution = open('sample_specificity.tsv', 'w')
stork = ''
for i in range(len(fastqs)):
    stork += fastqs[i] + '\t'
stork = stork[:-1]
stork += '\n'
contribution.write('anchor\tcompactor\t' + stork)
contribution.close()


# If we are not in a recursive call, take anchors from anchors_pvals.tsv file.
if sys.argv[3] != "recursive": 

    loading_anchors = time.time()
    print('Initializing parser.', flush=True)
   
    # Load summary file. 
    path = 'summary_filtered.tsv'
    
    # Optional filtering steps. Feel free to remove them! However, we recommend you keep the step which filters out UniVec.
    summary = pd.read_csv(path, engine='python', sep='\t',usecols=['anchor', 'effect_size_randCjs', 'mu_lev','anchor_top_ann','anchor_num_ann'])
    summary_take = summary[['anchor', 'effect_size_randCjs', 'mu_lev', 'anchor_top_ann','anchor_num_ann']].drop_duplicates(subset='anchor').sort_values(by='effect_size_randCjs', ascending=False, key=abs)
    #summary_take = summary_take[abs(summary_take['effect_size_randCjs']) >= 0.24]
    #summary_take = summary_take[summary_take['mu_lev'] >= np.percentile(list(summary_take['mu_lev']),80)]
    summary_take = summary_take[summary_take['anchor_top_ann'] != "anchor_hits_UniVec"]
    summary_take = summary_take[summary_take['anchor_top_ann'] != "anchor_hits_illumina_adapters"]
    #summary_take = summary_take[summary_take['number_nonzero_samples'] >= 10]
 
    # Take the union of the unannotated anchors and those anchors you've selected via filtering.
    no_annotations = list(summary_take[summary_take['anchor_num_ann'] == 0])
    sequences = list(summary_take['anchor'])
    sequences = list(set().union(sequences,no_annotations))
    
    
    # Write this list of anchors to a file. The parsing scripts will use this list to establish count dictionaries. 
    anchors = open('anchors.tsv', 'w')
    for sequence in sequences: 
        if sequence[0] in 'ACGT':
            anchors.write(sequence + '\n')
    anchors.close()


# Submit parallel jobs parsing samplesheet fastqs. 
for i in range(0, len(fastqs), fastqs_to_process_in_parallel):
    sbatch_file(",".join(fastqs[i:i+fastqs_to_process_in_parallel]), 'generate' + str(i), len(fastqs))
