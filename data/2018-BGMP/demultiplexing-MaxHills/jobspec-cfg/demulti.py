#!/usr/bin/env python3

import matplotlib.pyplot as plt
import gzip
import math
import numpy as np
def convert_phred(letter):
    """The function takes a single string character and converts it into the Illumina quality score."""
    phred_score = ord(letter)-33
    return phred_score
def avg_qs(fastq):
    """INPUT: fastq - A standard format fastq file of type str, readlen - The int length of the reads (in nucleotides)
    OUTPUT: A file containing a table of average q-scores at each position, A png file of the line-graph of average q-scores at each position"""
    with gzip.open(fastq, mode='rt') as f:
        # Initialize a line-count.
        LC = 0
        # Initialize a read-count.
        RC = 0
        # Iterate through each line in the file.
        for line in f:        
            # Increment the line count.
            LC += 1        
            # If the line number is a multiple of 4, do the following...
            if LC%4 == 0:
                line = line.strip()
                # Set the read length to the length of the line.
                readlen = len(line)
                # If this is the first read, intialize the array of zeroes, of size readlen, to hold qscores.
                if RC == 0:
                    all_qscores = np.zeros(readlen,dtype=int)
                # Increment the read-count.
                RC += 1
                # Initialize charIndex
                charIndex = 0
                for char in line:
                    # Get the decimal phred score at this position.
                    score = convert_phred(char)
                    # Store the score for the current position at its index within all_qscores. 
                    all_qscores[charIndex] += score
                    # Increment charIndex.
                    charIndex += 1
    # Store the average quality scores in their own array.
    mean_scores = all_qscores/RC
    # Make a template for the new file names.
    newfiles = fastq.split('/')[-1]
    # Create the filename for the Table.
    table = newfiles.replace('fastq','txt').replace('.gz','')
    # Create the filename for the Line Graph.
    graph = newfiles.replace('fastq','png').replace('.gz','')
    # Open the file to which we will print the table of averages.
    with open(table, 'w') as t:
        # Print a header for the columns.
        print("# Base Pos\tMean Quality Score",file=t)
        print("===========\t==================",file=t)
        for i in range(readlen):
            template = "{}\t\t{:0.2f}"
            out = str(template.format(i,mean_scores[i]))
            print(out,file=t)
    # Initialize empty lists to hold x,y axis labels.
    x = []
    y = []
    # Define the y axis in terms of the min and max values.
    ylow = math.floor(min(mean_scores))-2
    yhigh = math.ceil(max(mean_scores))+2
    # Create x labels stepwise by 2's.
    for n in range(0,readlen+1,2):
        x.append(n)
    # Create y labels stepwise by 1's.
    for n in range(ylow,yhigh):
        y.append(n)
    # Create the figure, with size and color specifications
    plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
    # Plot the mean_scores, with a dark blue line.
    plt.plot(mean_scores, "darkblue")
    # Create the x,y tick marks
    plt.xticks(x)
    plt.yticks(y)
    # Label the x,y axes.
    plt.xlabel("Position of Base in Read")
    plt.ylabel("Mean Quality Score")
    # Set the range for the x,y axes.
    plt.axis([-1,(readlen+1), ylow, yhigh])
    # Title the plot
    plt.title("Average Quality Scores @ Base Position X")
    # Show the graph grid, then save the figure.
    plt.grid(True)
    # Save the figure to the graph file.
    plt.savefig(graph,format='png')
# Call the function on the target files. 
avg_qs('/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R1_001.fastq.gz')  # This is Read 1.
avg_qs('/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R2_001.fastq.gz')  # This is Index 1.
avg_qs('/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R3_001.fastq.gz')  # This is Index 2.
avg_qs('/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R4_001.fastq.gz')  # This is Read 2. 
