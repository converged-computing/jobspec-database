#!/bin/bash
#SBATCH --job-name=pQuant_enc    # Job name
#SBATCH --partition=pe2   # Specify the partition
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks (MPI processes) per node
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --mem=40G                  # Memory per node (adjust as needed)
#SBATCH --mail-user=shong@nygenome.org   # Specify your email address

module add gcc/9.2.0
module add clang
module add cmake

# Navigate to the directory containing your executable
cd /gpfs/commons/groups/gursoy_lab/shong/Github/pQuant_enc/build

kmer_matrix_loc="../../pQuant_rust/kmer_matrix/kmer_$1_$2_$3.json"
# use matrix
# if $4 is true, use matrix(add argument --memory), else don't use matrix
echo "Slum job number = $SLURM_JOB_ID"
if [ "$4" = true ]; then
    ./pquant -t all -d $2 -k $1 -b -m $kmer_matrix_loc --debug_n_gene 5 --memory -e $3 
else
    ./pquant -t all -d $2 -k $1 -b --debug_n_gene 5 -e $3
fi
# ./pquant -t all -d $2 -k $1 -b --debug_n_gene 5 -e $3

 # don't use kmer matrix
# ./pquant -t all -d 5k -k 15 -sb -o ../slurm_save/$SLURM_JOB_ID --debug_n_gene 5
 