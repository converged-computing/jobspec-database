#!/projects/tau/packages/python/3.6.0/bin/python3
#Adrian Bubie
#9/9/17

#Quality and Index swapping: Our goal is to look through the lane of sequencing generated from your library preps and
#determine the level of index swapping and undetermined index-pairs, before and after quality filtering of index reads

#The program will take in the four fastq files, a qscore limit, and a text file containing all adapter indexes as inputs and filter each read
#based on the qscore limit. Because in our quality cutoff discernment step we determined a cutoff based on average quality per read, we will only throw out
#reads that have indexes with an *average* quality score below the cutoff (rather than throwing out a read if ANY of the nucleotides fall below threshold).
#Reads that pass filter are sorted by index pairs; reads that have indexes that are not among the possible index pairs 
#(a la mutation or index hopping) are sorted into an 'Unknown Index' dictionary item, and the rest are sorted into their respective index pair dictionary item.


## Argparse arguments - 4 fastq files (read1, read2, index1, index2), quality threshold, and index.tsv file
import argparse as ap

def get_arguments():
    parser = ap.ArgumentParser(description="Quality and Index Hopping filter")
    parser.add_argument("-q_cut", help="Quality score cut-off threshold for filtering reads", required=True, type=int)
    parser.add_argument("-r1", help="Filename for read1 fastq", required=True, type=str)
    parser.add_argument("-r2", help="Filename for read2 fastq", required=True, type=str)
    parser.add_argument("-i1", help="Filename for index1 fastq", required=True, type=str)
    parser.add_argument("-i2", help="Filename for index2 fastq", required=True, type=str)
    parser.add_argument("-indf", help="Text file containing all possible indexes", required=True)
    return parser.parse_args()

args = get_arguments()

# from the passed in arguments:
qcutoff = args.q_cut
indexes = args.indf

#First, we create a dictionary, and store all possible index pairs (order matters) as keys.
#The value for each key will be the number of counts that the index pair has been seen across all reads
ind_pairs = dict()

import itertools as it
with open(indexes,'r') as indx:
    sequences = []
    for line in indx:
        ind = line.split('\t')[1].strip('\n')
        
        sequences.append(ind)   #Grab all our indexes and put them in a list
    
    perms = it.product(sequences, repeat=2) #Create a product of all possible index pairs (including the index with itself)
    for p in perms:
        ind_pairs[p] = 0  # Initialize each pair as a key in the dict and the value to 0

# Additionally, let's add a dictionary key for our reads that do not have a possible index pair (for the reads with 'N' or that have had
# a point mutation in the index that may push it out of the possible pairs).
ind_pairs['Unknown'] = 0

#Next, we both quality filter and sort our sequencing reads/indexes into files baised on the index pairing of each read.
#We start by defining a function that will read in lines from our files in set of 4, such that we capture only one fastq entry at a time:     
def four_line_grouper(file, n):
    read = it.islice(file, n)
    return read

r1 = open(args.r1,'r')
r2 = open(args.r2,'r')
i1 = open(args.i1,'r')
i2 = open(args.i2,'r')

for i in range(0,363246734):
    read1 = [x for x in four_line_grouper(r1,4)]
    read2 = [x for x in four_line_grouper(r2,4)]
    index1 = [x for x in four_line_grouper(i1,4)]
    index2 = [x for x in four_line_grouper(i2,4)]
    
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} #Complement dict for transforming index 2 to reverse complement
    keep1 = True  #Flags for keeping the read, one for each index
    keep2 = True

    ind1_qual_ave = 0 # Set a value to store the quality score average for each index
    ind2_qual_ave = 0 

    for char in index1[3].strip('\n'):  #Loop through every value in the quality string for index1
        qual = (ord(char)-33)
        ind1_qual_ave += qual
        
    for char in index2[3].strip('\n'):  #Loop through every value in the quality string for index2
        qual = (ord(char)-33)
        ind2_qual_ave += qual
   
    ind1_qual_ave = ind1_qual_ave/8  # Divide by the number of nucleotides in the index
    ind2_qual_ave = ind2_qual_ave/8

    if ind1_qual_ave < qcutoff:
            keep1 = False    #If the index ave qual falls below the quality cutoff, flip the quality flag
            #print(ind1_qual_ave)

    if ind2_qual_ave < qcutoff:
            keep2 = False    #If the index ave qual  falls below the quality cutoff, flip the quality flag
 
    if keep1 == True and keep2 == True:               #Check to see both indexes passed quality check
        ind1 = index1[1].strip('\n')
        ind2 = index2[1].strip('\n')
        
        if 'N' not in ind2:                     # Check to see if the index has N; if it does, we don't need to take reverse compliment because we already know it won't match known index pair.
            ind2 = ind2[::-1]                        #Need to make the reverse complement of the second index
            ind2 = list(ind2)
            for i in range(0,len(ind2)):
                ind2[i] = complement[ind2[i]]

            ind2 = ''.join(ind2)
        
        index_pair = (ind1,ind2)
        #print(index_pair)
        
        
        if index_pair in ind_pairs:   #If the read has passed quality, and is in the possible list of index pairs:
            
            ind_pairs[index_pair] += 1 #if the index pair of this is in the dictionary of index pairs, increment the count for that pairing
        
        
        elif index_pair not in ind_pairs:   #If the read has passed quality but does not contain a possible index pair:
            
            ind_pairs['Unknown'] += 1  #Increment the count of the 'Unknown' item, for reads with indexes outside the possible set

r1.close()
r2.close()
i1.close()
i2.close()

print("Total number of records that passed quality filtering:",sum(ind_pairs.values()))
out = open(('Index_Pair_Counts_'+str(qcutoff)+'.txt'),'w')
for item in ind_pairs.items():
    out.write(str(item[0])+'\t'+str(item[1])+'\n')

out.close()
