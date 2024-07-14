import os
import RNA
from Bio import SeqIO
from multiprocessing import Pool, cpu_count

OUTPUT_FILE = "results.txt"
DATA_PATH = '240226_partial_sequences.fa'

def process_sequence(record):
    seq_id = record.id
    sequence = str(record.seq)
    fc = RNA.fold_compound(sequence)
    (ss, mfe) = fc.mfe()
    result = f'Sequence ID: {seq_id}\nSequence: {sequence}\nSecondary Structure: {ss}\nMinimum Free Energy (MFE): {mfe:.2f}\n\n'
    return result

def main():
    with open(OUTPUT_FILE, 'w') as f:
        sequences = SeqIO.parse(DATA_PATH, "fasta")
        with Pool(processes=cpu_count()) as pool:
            for result in pool.imap(process_sequence, sequences, chunksize=10):
                f.write(result)

if __name__ == "__main__":
    main()

