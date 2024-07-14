#!/bin/bash

# Define the download function with one line from the tsv file as the feed-in
prokka_data(){
    info=($1)
    ASVtag=${info[0]} # ASV tag, e.g. c497da3b39f30aceede6bec3b03cd100
    str2=${info[1]} # genome id, e.g. GB_GCA_905618805.1

    extline="$(grep -m1 -i "$ASVtag" TaxInfo.tsv)"

    IFS=';' read -ra ADDR <<< "$extline"
    kingdom="$(echo ${ADDR[0]} | cut -d' ' -f2)"
    # echo $kingdom

    # echo $str1 
    # echo $str2
    prokkaoutput="prokkaP16NS"

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
        if [[ $kingdom == *"Bacteria"* ]]; then
            singularity exec prokka.sif prokka --norrna --notrna --centre X --compliant --kingdom Bacteria $genome_file2 --prefix $str2 --outdir $prokkaoutput/$str2 #--prefix $prokkaoutput/$str2 
        elif [[ $kingdom == *"Archaea"* ]]; then
            singularity exec prokka.sif prokka --norrna --notrna --centre X --compliant --kingdom Archaea $genome_file2 --prefix $str2 --outdir $prokkaoutput/$str2 #--prefix $prokkaoutput/$str2 
        else
            singularity exec prokka.sif prokka --norrna --notrna --centre X --compliant --kingdom Bacteria $genome_file2 --prefix $str2 --outdir $prokkaoutput/$str2 #--prefix $prokkaoutput/$str2 
        fi

    fi
}
export -f prokka_data

echo "Prokka Starts"

# The second argument specifies the input file
inputfile=$2

# parallel the download function with the first argument specifying number of cores
cat $inputfile | parallel -j $1 prokka_data

echo "Done"

