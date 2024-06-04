#!/bin/sh -l

#SBATCH --job-name evolinc 
#SBATCH -A meixiazhao
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2 
#SBATCH --time=7-0:00
#SBATCH -o /blue/meixiazhao/lee.gwonjin/Soybean_project/scripts/outtext/evolinc_TE.out
#SBATCH -e /blue/meixiazhao/lee.gwonjin/Soybean_project/scripts/outtext/evolinc_TE.err
#SBATCH --mail-type=all          
#SBATCH --mail-user=lee.gwonjin@ufl.edu

# Use Cyverse if this is not working https://drive.google.com/file/d/0B-ferWixi_V3cmh0QzhJeXRXSE0/view?resourcekey=0-v-2R4wUvw83njM5CAu7C-Q

cd /blue/meixiazhao/lee.gwonjin/Soybean_project/lncRNA/evolinc

module load singularity

#Make tmp folder if necessary

#Remove a previous sif file
rm evolinc-i_1.7.5.sif

# Pull the image from CyVerse Dockerhub
singularity pull docker://evolinc/evolinc-i:1.7.5

# See the command line help for the Docker image
#singularity run docker://evolinc/evolinc-i:1.7.5 -h 

# Download some sample data 
#git clone https://github.com/Evolinc/Evolinc-I.git

# Run Evolinc-I With mandatory files
singularity run docker://evolinc/evolinc-i:1.7.5 \
-c /blue/meixiazhao/lee.gwonjin/Soybean_project/lncRNA/mapping/assembly/merged_asm_ref/Transctipts64Assembly.gtf \
-g /blue/meixiazhao/lee.gwonjin/Soybean_project/reference/genome/PhytozomeV13/Gmax/Wm82.a4.v1/assembly/Gmax_508_v4.0.fa \
-u /blue/meixiazhao/lee.gwonjin/Soybean_project/reference/genome/PhytozomeV13/Gmax/Wm82.a4.v1/annotation/Gmax_508_Wm82.a4.v1.gene.gff3 \
-b /blue/meixiazhao/lee.gwonjin/Soybean_project/reference/genome/PhytozomeV13/Gmax/TE/TE_seqforEvolinc/TE_sequences.fa \
-o evolinc_out \
-n 4



# INFO:    Creating SIF file...

# Usage : sh /evolinc_docker/evolinc-part-I.sh -c cuffcompare -g genome -u user_gff -o output -n threads [-b TE_RNA] [-t CAGE_RNA] [-x Known_lincRNA]


#  -c </path/to/cuffcompare output file>

#  -g </path/to/reference genome file>

#  -u </path/to/user reference annotation file>

#  -r </path/to/reference annotation file> # This option is specific to CyVerse Discovery Environment

#  -o </path/to/output file>

#  -n <number of threads>

#  -b </path/to/Transposable Elements file>

#  -t </path/to/CAGE RNA file>

#  -x </path/to/Known lincRNA file>

#  -h Show this usage information
