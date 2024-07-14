import numpy as np
import pandas as pd
import os
import gzip
import random
import time
import sys
from Bio import SeqIO

def make_intermediaries(fastq,  max_anchor_reads=200, anchor_length=27):

    if sys.argv[3] != "recursive":
        anchor_list = pd.read_csv('anchors.tsv', engine='python', sep='\t', header=None)
        anchor_list = list(anchor_list.iloc[:,0])

    if sys.argv[3] == "recursive":
        anchor_list = pd.read_csv('anchors_recursive.tsv', engine='python', sep='\t', header=None)
        anchor_list = list(anchor_list.iloc[:,0])
    
    # Construct anchor dictionaries.
    zeroes = [0 for i in range(len(anchor_list))]
    count_dictionary = dict(zip(anchor_list, zeroes))
    empty_strings = [[] for i in range(len(anchor_list))]
    read_dictionary = dict(zip(anchor_list, empty_strings))

    # Utility.
    num_anchors = len(anchor_list)

    # Set threshold for reads per fastq: default to 100 million / number of samples.
    num_reads = 1e6

    # Create variables for performance tracking.
    local_reads = 0
    written_reads = 0
    start_time = time.time()

    # Create dictionary for per-sample anchor count thresholding.
    fastq_specific_count = dict(zip(anchor_list, zeroes))

    # Open and access records using BioPython.
    with gzip.open(fastq, mode="rt") as handle:
        for record in SeqIO.parse(handle, 'fastq'):

            # Break if we've exceeded the number of reads we wish to visit.
            # Break if we've fully populated each of our anchor-read lists. 
            if local_reads >= num_reads or written_reads == num_anchors * max_anchor_reads:
                break

            # If we haven't exceeded our read bounds, proceed and increment counter.
            if not local_reads >= num_reads and not written_reads == num_anchors * max_anchor_reads:
                local_reads += 1
                read = str(record.seq)

                # Index only to the read_length - 50th location in the read.
                for i in np.arange(len(read) - 50):

                    # Establish anchor so as to avoid slicing it from read repeatedly.
                    anchor = read[i:i+anchor_length]

                    # Check if we've exceeded count for an anchor.
                    # This simultaneously allows us to pass a read window if it's not an anchor.
                    if anchor in fastq_specific_count:
                    
                        # Increment the count of this anchor in this sample.
                        fastq_specific_count[anchor] = fastq_specific_count[anchor] + 1

                        # If we are at or below our count limit:
                        if fastq_specific_count[anchor] <= max_anchor_reads:

                            # Add read to dictionary and increment sample-specific written read count. 
                            read_dictionary[anchor].append(read[i:] + '\t' + fastq + '\n')
                            written_reads += 1
     
        # Log filename, reads, writes, and execution time.
        with open('execution_time.tsv', 'a') as write_log:
            end_time = time.time() - start_time
            write_log.write(fastq + '\t' + str(local_reads) + '\t' + str(written_reads) + '\t' + str(end_time) + '\n')

    # Write the resulting read lists to the appropriate intermediate files.
    for key in read_dictionary.keys():
        with open(key + '.intermediary', 'a') as anchor_log:
            for item in read_dictionary[key]:
                anchor_log.write(item)
        anchor_log.close()

    return

fastqs = sys.argv[1].split(",")

for fastq in fastqs:
    make_intermediaries(fastq)
