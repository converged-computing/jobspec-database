#!/usr/bin/env bash

#PBS -N qsmooth-parallel-submit
#PBS -l select=1:ncpus=1:mem=10GB
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -P personal
#PBS -q normal

# load module
module load singularity

# set variables
INPUT_DIR = "/home/users/ntu/rwu008/scratch/inputs"
OUTPUT_DIR = "/home/users/ntu/rwu008/scratch/data/outputs/top5"

# find all parquet files just within the INPUT_DIR and map them to the file_paths array 
readarray -t file_paths < <(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -name "*.parquet" -type f)

echo "The total number of parquet files:" "${#file_paths[@]}"

# initialize the arrays
smallfiles_arr=()
bigfiles_arr=()

# split the filepath array into two groups
for f in "${file_paths[@]}"; do
    # find the file size
    file_size=$(stat -c%s $f)

    # check if the file size is below 1000000000
    if [ $file_size < 1000000000 ]; then
        smallfiles_arr+=("$f") 
    else
        bigfiles_arr+=("$f")
done

# print array lengths
echo "Small file array: " "${#smallfiles_arr[@]}"
echo "Big file array: " "${#bigfiles_arr[@]}"

# initialize count
count=0
# process small files first
for file in "${smallfiles_arr[@]}"; do
    # increment count
    ((count++))
    # print current file
    echo "File no $count: $file"
    
    # check if output file exists
    file_name=$(basename $file)
    output_file_prefix="${file_name%%.parquet}-5processed.parquet"


    if [ -f "${OUTPUT_DIR}/${output_file_prefix}" ]; then
        echo "${OUTPUT_DIR}/${output_file_prefix} exists. Skipping file..."
        continue
    else
        echo "${OUTPUT_DIR}/${output_file_prefix} does not exist. Submitting job..."
        if [ $count -le 90 ]; then
            # submit the file as a job
            qsub -v PARQUETFILE="${file}" path/to/qsmooth-submit.pbs
        elif [ $count -eq 91 ]; then
            sleep 1h
            continue
        else
            if [[ $(( count % 45 )) -ne 0]]; then
                # submit the file as a job
                qsub -v PARQUETFILE="${file}" path/to/qsmooth-submit.pbs
            else
                sleep 1h
                continue
            fi
        fi
    fi
done

# reinitialize count
count=0
# process big files
for file in "${bigfiles_arr[@]}"; do
    # increment count
    ((count++))
    # print current file
    echo "File no $count: $file"
    
    # check if output file exists
    file_name=$(basename $file)
    output_file_prefix="${file_name%%.parquet}-5processed.parquet"

    if [ -f "${OUTPUT_DIR}/${output_file_prefix}" ]; then
        echo "${OUTPUT_DIR}/${output_file_prefix} exists. Skipping file..."
        continue
    else
        echo "${OUTPUT_DIR}/${output_file_prefix} does not exist. Submitting job..."
        if [ $count -le 90 ]; then
            # submit the file as a job
            qsub -v PARQUETFILE="${file}" path/to/qsmooth-submit.pbs
        elif [ $count -eq 91 ]; then
            sleep 24h
            continue
        else
            if [[ $(( count % 45 )) -ne 0]]; then
                # submit the file as a job
                qsub -v PARQUETFILE="${file}" path/to/qsmooth-submit.pbs
            else
                sleep 24h
                continue
            fi
        fi
    fi
done

echo "Exiting script..."





