#!/bin/bash

#SBATCH -p milanq # partition (queue)
#SBATCH -t 0-20:00 # time limit (D-HH:MM)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

export PATH=$PATH:/usr/local/cuda/bin

#nvidia-smi

#nvcc --version
make 

# Set the LDLIBRARY environment variable with the specified prefix
LD_LIBRARY_PATH="/home/santiago/local/lib"
export LD_LIBRARY_PATH

# Ensure the execute.out executable is in the same directory as this script
EXECUTABLE="./execute"

if [ ! -x "$EXECUTABLE" ]; then
    echo "The 'execute.out' executable does not exist in the current directory or is not executable."
    exit 1
fi

# List of input matrices
INPUT_MATRICES=("AG-Monien/wave.mtx.gz"
    "AG-Monien/debr.mtx.gz"
    "AMD/G3_circuit.mtx.gz"
    "Andrianov/ins2.mtx.gz"
    "Andrianov/mip1.mtx.gz")
    

mkdir -p /work/$USER/tmp1

# Define the path to the directory containing the matrices
MATRIX_DIR="/global/D1/homes/james/sparcity/suitesparse/mtx/original/" 
OUTPUT_DIR="/global/D1/homes/santiago/reorderings/rabbit/"   

for input_matrix in "${INPUT_MATRICES[@]}"; do
    # Extract the matrix name and the group name
    matrix_name=$(basename "${input_matrix}" .gz)
    group_name="${input_matrix%%/*}"

    # Concatenate path with matrix name
    path_to_input_matrix="$MATRIX_DIR$input_matrix"

    # Decompress the input matrix
    gunzip -c "$path_to_input_matrix" > "/work/$USER/tmp1/$matrix_name"

    # Run the program with the input matrix
    "$EXECUTABLE" "/work/$USER/tmp1/$matrix_name" "1"

    # Create a folder with the group name if it doesn't exist
    mkdir -p "$OUTPUT_DIR$group_name"

    # Rename the generated output to the desired name and desired folder
    mv "order1.mtx" "$OUTPUT_DIR$group_name/$matrix_name"

    # Compress the output file to a .gz file
    gzip -c "$OUTPUT_DIR$group_name/$matrix_name" > "$OUTPUT_DIR$group_name/$matrix_name.gz"

    # Remove the original .mtx file to free up space
    rm "$OUTPUT_DIR$group_name/$matrix_name"
    rm "/work/$USER/tmp1/$matrix_name"

    echo "Processed $input_matrix and saved as $output_file"
done


