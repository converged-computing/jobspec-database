#!/bin/bash

# Define the download function with one line from the tsv file as the feed-in
prokka_data(){
    info=($1)
    str2=${info[1]} # genome id, e.g. GB_GCA_905618805.1
    # echo $str1 
    # echo $str2
    prokkaoutput="prokkaooutput"

    if [[ ! -e $prokkaoutput/$str2 ]]; then
        # create folders to hold annotated genomes for $str2
        # mkdir -p $prokkaoutput
        # read json file
        json_file="GetGenome/genomes/${str2}/ncbi_dataset/data/dataset_catalog.json"
        
        # get the path to the genome file
        genome_file1=$(jq -r .assemblies[1].files[0].filePath ${json_file})
        # e.g. /GetGenome/genomes/GB_GCA_902623215.1/ncbi_dataset/data/GCA_902623215.1/GCA_902623215.1_AG-893-M16_genomic.fna
        genome_file2="GetGenome/genomes/${str2}/ncbi_dataset/data/${genome_file1}"
        
        # download data using datasets from ncbi
        singularity exec prokka.sif prokka $genome_file2 --prefix $str2 --outdir $prokkaoutput/$str2 #--prefix $prokkaoutput/$str2 

    fi
}
export -f prokka_data

echo "Prokka Starts"

# The second argument specifies the input file, e.g. 221117-1910.P16N-S.16S.dna-sequences.tsv
inputfile=$2

# parallel the download function with the first argument specifying number of cores
cat $inputfile | parallel -j $1 prokka_data

echo "Done"

