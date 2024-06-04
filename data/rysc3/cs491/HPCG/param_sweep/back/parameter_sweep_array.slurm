#!/bin/bash

#SBATCH --job-name HPCG_params
#SBATCH --partition condo
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus 1
#SBATCH --array 1-2
#SBATCH --mail-user rscherbarth@unm.edu
#SBATCH --mail-type all

# Load the required packages (gcc 11 and HPL)
# module load gcc/11.2.0-655h hpl

# Load modules for hpcg on cpu:
# module load intel/20.0.4 hpcg 

# Load modules for singularity (gpu version)
module load singularity

export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_NUM_THREADS=64

# Set a place to record the results
RESULTS_FILE=$SLURM_SUBMIT_DIR/HPCG_results.csv

PARAMS_FILE=$SLURM_SUBMIT_DIR/HPCG_params.csv
TEMPLATE_FILE=$SLURM_SUBMIT_DIR/HPCG_template.dat

# Check for errors
if test -f $PARAMS_FILE; then
    echo Using parameter file $PARAMS_FILE
else
    echo Error $PARAMS_FILE not found
    exit 1
fi

if test -f $TEMPLATE_FILE; then
    echo Using template file $TEMPLATE_FILE
else
    echo Error $TEMPLATE_FILE not found
    exit 2
fi
    
# Get the Nth line from our parameter file - where N is the array ID
PARAMS=$(head -n $SLURM_ARRAY_TASK_ID $PARAMS_FILE | tail -n 1)
echo Read param line $SLURM_ARRAY_TASK_ID: $PARAMS

# Get the Xth element of that line (comma separated)
NX_VAL=$(echo $PARAMS | awk -F"," '{print $1}') # awk: Aho, Weinberger, and Kernighan
echo Read param NX: $NX_VAL
NY_VAL=$(echo $PARAMS | awk -F"," '{print $2}') # awk: Aho, Weinberger, and Kernighan
echo Read param NY: $NY_VAL
NZ_VAL=$(echo $PARAMS | awk -F"," '{print $3}') # awk: Aho, Weinberger, and Kernighan
echo Read param NZ: $NZ_VAL
TIME_VAL=$(echo $PARAMS | awk -F"," '{print $4}') # awk: Aho, Weinberger, and Kernighan
echo Read param TIME: $TIME_VAL


# Create a new working directory for each instance of xhpl since it needs it expects it's own HPL.dat
SCRATCH_DIR=/carc/scratch/users/$USER

# Make a temporary directory for our work - we will delete this at the end
TMP_DIR=$(mktemp --directory -p $SCRATCH_DIR)
echo Temp directory: $TMP_DIR

# Make a subdirectory with the SLURM array task id to make debugging easier
TMP_WORKING_DIR=$TMP_DIR/$SLURM_ARRAY_TASK_ID
mkdir -p $TMP_WORKING_DIR
echo Created temporary working directory: $TMP_WORKING_DIR

# Make the new working directory the current directory so xhpl runs in there
cd $TMP_WORKING_DIR
echo Now running in $PWD

# Substitute the parameter value read from the PARAMS file above into the HLP.dat for this instance
# Copy the template file into the temp working directory
cp $TEMPLATE_FILE $TMP_WORKING_DIR/hpcg.dat # @ryan

# Substitute the new block size in for the string <BLOCK_SIZE>
sed -i "s/<NX_VAL>/$NX_VAL/g" hpcg.dat # sed: string edit
sed -i "s/<NY_VAL>/$NY_VAL/g" hpcg.dat # sed: string edit
sed -i "s/<NZ_VAL>/$NZ_VAL/g" hpcg.dat # sed: string edit
sed -i "s/<TIME_VAL>/$TIME_VAL/g" hpcg.dat # sed: string edit
# @ryan 

echo Running xhpl in $TMP_WORKING_DIR...
srun --mpi=pmi2 xhpl
echo xhpl finished

# The HPL.dat file tells xhpl to write to HPL.out.
# Extract the throughput with grep and awk

# 1. Find the line containing Gflops and print it and the following line
RESULT_HEADER_AND_DATA_LINES=$(grep --after 2 Gflops HPL.out)
echo Results: $RESULT_HEADER_AND_DATA_LINES

# 2. Get just the data line
RESULT_DATA_LINE=$(echo $RESULT_HEADER_AND_DATA_LINES | tail -n 1)
echo Results data: $RESULT_DATA_LINE

# 3. Get the last field in the data line, that's the Gigaflops.
GFLOPS=$(echo $RESULT_DATA_LINE | awk -F" " '{print $NF}')
echo Results Gflops: $GFLOPS 

echo Writing input parameters and gflops to $RESULTS_FILE
echo $NX_VAL, $NY_VAL, $NZ_VAL, $TIME_VAL, $GFLOPS >> $RESULTS_FILE

# Clean up the temporary working directory
rm -r $TMP_DIR
echo Deleted $TMP_DIR
