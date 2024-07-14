import os
import sys
from Bio import SeqIO

def getNonPlasmids(file, speciesPath, coverage):
    # Open fasta file
    records = []
    genome_size = 0
    for record in SeqIO.parse(file, 'fasta'):
        if 'plasmid' not in record.description or 'Plasmid' not in record.description:
            records.append(record)
            genome_size += len(record.seq)
    # Create fasta file with no plasmids
    SeqIO.write(records, os.path.join(speciesPath, 'genomeWOplasmids.fna'), "fasta")
    size_gbp = round(int(coverage)*genome_size/1000000000, 5)
    print(size_gbp)

if __name__ == '__main__':
    getNonPlasmids(sys.argv[1], sys.argv[2], sys.argv[3])