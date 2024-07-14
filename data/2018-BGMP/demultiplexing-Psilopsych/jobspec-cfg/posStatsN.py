#!/usr/bin/env python3

#Gzip module used to process zipped files
import gzip

numOfPos = 8
mean_scores =[0.0] * numOfPos


def convert_phred(letter):
    """Converts a single character into a phred score. Ord function does this """
    phred = ord(letter) - 33 #Phred 33 encoding
    return phred


def populate_array(file):
    i = 0
    LN = 0
    scores = []
    for i in range(numOfPos):
        scores.append(0.0)
    #gzip used to process file while still zipped. 't' used as parameter for zipped files also
    with gzip.open(file, 'rt') as fq:
        for line in fq:
            line = line.strip("\n")
            i+=1
            LN += 1
            if i % 4 == 3:  #Quality score lines
                for k in range(len(line)):
                    score = convert_phred(line[k]) 
                    scores[k] += score
    return(scores, LN)

                                
file = "../../../shared/2017_sequencing/1294_S1_L008_R3_001.fastq.gz"
#file = "testR2.txt.gz"
mean_scores, NR = populate_array(file)


for k in range(len(mean_scores)):
    mean_scores[k] = float(mean_scores[k]) / float((NR / 4))

print('# Base Pair\t Mean Quality Score')
for k in range(len(mean_scores)):
    print(k, '\t', mean_scores[k])

