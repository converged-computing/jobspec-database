#!/bin/bash
#SBATCH --job-name="70_75"
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --partition amd
#SBATCH --time=70:00:00

DIR="/gpfs/space/home/alihakim/blast/blast_70_75"  # Update with the path to your fasta files
HEADER="qseqid stitle qlen slen qstart qend sstart send evalue length nident mismatch gapopen gaps sstrand qcovs pident"

# Process each fasta file in the directory
for fasta_file in "$DIR"/*.fasta; do
    # Extract the basename of the fasta file
    BASENAME=$(basename "$fasta_file" .fasta)

    # Create a directory for each fasta file
    mkdir -p "$DIR/$BASENAME"

    TOTAL_SEQS=$(grep -c '^>' "$fasta_file")

    if (( TOTAL_SEQS <= 500 )); then
        NUM_CHUNKS=5
    else
        NUM_CHUNKS=10
    fi

    SEQ_PER_CHUNK=$((TOTAL_SEQS / NUM_CHUNKS))

    # Split the FASTA file
    awk -v prefix="$DIR/$BASENAME/${BASENAME}_chunk_" -v chunk_size="$SEQ_PER_CHUNK" '/^>/{n++;if(n%chunk_size==1) {m++; close(f); f=prefix m ".fasta"}} {print > f}' "$fasta_file"

    for file in "$DIR/$BASENAME/${BASENAME}_chunk_"*.fasta; do
        (
            blastn -query $file \
            -db /gpfs/space/home/alihakim/analysis/database/UNITE \
            -word_size 7 \
            -task blastn \
            -num_threads 12 \
            -outfmt "6 delim=+ $HEADER" \
            -evalue 0.001 \
            -strand both \
            -max_target_seqs 10 \
            -max_hsps 1 \
            -out "${file%.fasta}_blast_results.txt"

            # Add header to the output file
            sed -i '1i'"$HEADER" "${file%.fasta}_blast_results.txt"

            rm $file
        ) &
    done
    wait

    # Combine and deduplicate the results
    cat "$DIR/$BASENAME/${BASENAME}_chunk_"*_blast_results.txt > "$DIR/$BASENAME/combined_blast_top10hit.txt"
    awk -F '+' '!seen[$1]++' "$DIR/$BASENAME/combined_blast_top10hit.txt" > "$DIR/$BASENAME/${BASENAME}.txt"
    rm "$DIR/$BASENAME/${BASENAME}_chunk_"*_blast_results.txt