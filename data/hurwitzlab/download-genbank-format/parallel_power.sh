#!/bin/bash

#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=2:mem=3gb
#PBS -l walltime=72:00:00
#PBS -l cput=72:00:00
#PBS -M scottdaniel@email.arizona.edu
#PBS -m ea
#PBS -o out
#PBS -j oe

echo Started $(date)

export WD=$PBS_O_WORKDIR

cd $WD
unset module
set -u

TMP_FILES=$(mktemp)

get_lines $TODO $TMP_FILES $PBS_ARRAY_INDEX $STEP_SIZE

NUM_FILES=$(lc $TMP_FILES)

if [[ $NUM_FILES -lt 1 ]]; then
    echo Something went wrong or no files to process
    exit 1
else
    echo Found \"$NUM_FILES\" files to process
fi

for file in $(cat $TMP_FILES); do

	echo "Doing $(basename $file)"

    unzipped=$(basename $file .gz)

    #need to do this because of mistake
    rm "$unzipped".fasta

    gunzip $file
    $PRJ_DIR/genbank_to_fasta.py -i $unzipped -s nt

done

echo Finished $(date)
