#!/bin/sh
#SBATCH --job-name=iqtree
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10gb
#SBATCH --time=100:00:00
#SBATCH --output=logs/iqtree.plastid-%j.out
#SBATCH --error=logs/iqtree.plastid-%j.err


cd $SLURM_SUBMIT_DIR
date
hostname

module purge
module load cuda/9.2.88 intel/2018.1.163 openmpi/3.1.2 iq-tree/1.6.10

#cd ../mafft/

#files=( $(ls *_alignment.fasta) )
#file="${files[$SLURM_ARRAY_TASK_ID]}"

# -s alignment file
# -te fixed user tree
# -o outgroup
# -nt number of CPUs
# -m Model selection (TEST = automatic model selection follwed by tree building using model)
# -merit select optimality criterion (AIC, AICc, BIC)
# -q specify partition file OR -spp partitions have different evoltuionary speed OR -sp unlinked partitions
# -g specify a topological constraint, does not need all taxa
# -pers perturbation strength, recommended a low number for small sequences
# -nstop number of unsucessful iteration, again, recommended for small sequences
# -bb ultrafast bootstrap iterations
#-spp Trimal90_noninterleaved_supercontig_partition_file.txt

#echo "running IQ-Tree on ${file}"

#iqtree -s ${file} -nt $SLURM_CPUS_ON_NODE -seed $RANDOM -m TEST -merit BIC -pers 0.2 -nstop 500 -bb 1000 > ../iqtree/${file}.tre

iqtree -s plastid_whole_read_mapped_reduced.fasta -nt $SLURM_CPUS_ON_NODE -seed $RANDOM -m TEST -merit AIC -bb 1000
