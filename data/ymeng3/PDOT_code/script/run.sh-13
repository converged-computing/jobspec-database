#!/bin/bash
#SBATCH --job-name=pdot_dataset1
#SBATCH --output=/home/ymeng3/result/result13.txt
#SBATCH --error=/home/ymeng3/result/error13.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --account=pi-haihaolu

echo "Starting run.sh-13..."

# pdotans dataset2 100: 1-10
module load julia  # Load Julia module, if available 
julia /home/ymeng3/experiments/code/pdot_code_ans/test/dataset2.jl


# Check if the Julia script ran successfully
if [ $? -eq 0 ]; then
  echo "Julia script completed successfully."
else
  echo "Julia script encountered an error. Check the logs for details."
fi

# Print a message to indicate that the script is finished
echo "run.sh-13 completed."