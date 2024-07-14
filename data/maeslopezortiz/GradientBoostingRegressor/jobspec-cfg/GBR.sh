#!/bin/sh

# Slurm directives:
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --time 1-5:10
#SBATCH --output=logs/GradientBoostingRegressor.out
#SBATCH --error=logs/GradientBoostingRegressor.err
#SBATCH --job-name=GradientBoostingRegressor

### How to run this script:
# Verify all your files are in the same directory
# Submit it to the cluster using Slurm with: sbatch GBR.sh species_name trait_value
#multiple species
 #for i in "${!SPECIES[@]}"; do sbatch GBR.sh "${SPECIES[i]}" "${TRAIT[i]}"; done
# Make sure to load the appropriate Python module, e.g., module load python/3.7.4 if you are in euler
# Download the EasyGeSe folder from the repository 
#module load python/3.7.4
#module load gcc/6.3.0 r/4.1.3
#git clone https://github.com/stevenandrewyates/EasyGeSe

SPECIES=$1
TRAIT=$2

# This file contains species names and the number of traits
mkdir GBR_results #generate the directory where the results will go.

python_success=true

for TRAIT in $(seq 1 $TRAIT); do
   for CV in {1..5}; do
      for FOLD in {1..2}; do
         echo "Running with arguments: $SPECIES, TRAIT=$TRAIT, CV=$CV, FOLD=$FOLD"
         python GBR_na.py "$SPECIES" "$TRAIT" "$CV" "$FOLD" # To execute your Python script. It can be modified to run for testing estimators or number of samples
         python_exit_status=$? # Capture the exit status of the Python script

         if [ $python_exit_status -ne 0 ]; then
            python_success=false
            echo "Python script execution failed."
            break 3  # Exit the outer loop if Python script fails
         fi
      done
   done
done

if $python_success; then
   echo "Python Script execution completed."
fi
