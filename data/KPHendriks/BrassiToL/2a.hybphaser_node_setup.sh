#!/bin/bash

# When running on Stampede2 cluster, detail the following:
#   *** Serial Job on Normal Queue ***
#SBATCH	-A TG-TRA170019	                # Account to charge for this run
#SBATCH -J HybPhaser         								# Job name
#SBATCH -o HybPhaser.o%j     								# Name of stdout output file
#SBATCH -e HybPhaser.e%j   									# Name of stderr error file
#SBATCH -p normal          							# Queue (partition) name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=67
#SBATCH -t 48:00:00        							# Run time (hh:mm:ss)
#SBATCH --mail-user=kasper.hendriks@naturalis.nl
#SBATCH --mail-type=none    						# Send no emails

:"
#Load needed versions of modules.
module load intel/17.0.4
module load gnuparallel
module load biocontainers/0.1.0 	#With this module, various other tools/modules can be loaded
module load bwa/0.7.16a
module load blast/2.6.0
module load samtools/1.10
module load python2
module list
pwd
date


##Run HybPhaser step 1 first for the Nikolov2019 data.

#Start by running, in parallel (for all samples), the following script that copies mapped data to a HybPhaser-specific directory,
#and subsequently runs step 1 of HybPhaser: generate consensus sequences.
#See for more information on the use of HybPhaser: https://github.com/LarsNauheimer/HybPhaser
parallel -j 2 bash 2b.hybphaser_sample_Nikolov2019.sh {} :::: namelist.txt

##Run HybPhaser step 1 second for the Angiosperms353 data.

#Start by running, in parallel (for all samples), the following script that copies mapped data to a HybPhaser-specific directory,
#and subsequently runs step 1 of HybPhaser: generate consensus sequences.
#See for more information on the use of HybPhaser: https://github.com/LarsNauheimer/HybPhaser
parallel -j 2 bash 2b.hybphaser_sample_Angiosperms353NewTargets.sh {} :::: namelist.txt

##Now combine results from HybPhaser, sample by  sample, so that we can assess SNPs for all nuclear genes at a time.
#This script includes the removal of 36 genes from the Nikolov2019 set, because these overlap with genes from the Angiosperms353 set.
parallel -j 2 bash 2c.hybphaser_combine_results_bait_sets.sh {} :::: namelist.txt
"
#Move to the directory with combined results.
cd results_HybPhaser_allGenes/

#Reload needed versions of modules.
module load intel/18.0.0
module load impi/18.0.0
module load RstatsPackages/3.5.1
module load Rstats/3.5.1
module load gnuparallel
module list
pwd
date

#And now run step 1 of HybPhaser scripts to count SNPS and detect paralogs.
#Note that the original HybPhaser R script 1a was cut into two, such that the first part can be run in parallel,
#which saves a lot of time when handling hundreds of samples.
#Furthermore, to avoid studying 36 duplicate genes among the two baits sets used, script 1a calls a target reference file from which
#these 36 genes for the Nikolov2019 kit have been removed.

#parallel -j 5 Rscript ../hybphaser_scripts_updated_for_BrassiWood_project/1a_count_snps_perSample.R config_inclusive.txt :::: ../namelist.txt

#Rscript ../hybphaser_scripts_updated_for_BrassiWood_project/1a_count_snps_combineResults.R config_inclusive.txt

#Rscript ../hybphaser_scripts_updated_for_BrassiWood_project/1b_assess_dataset.R config_inclusive.txt

#Collect consensus and contig reads using HybPhaser script 1c.
#I cut this into three: one general script to set up directories, then a script to be run in parallel to collect data by locus.
#The third script removes loci (and samples from loci) as based on paralog calls from script 1b.
#(I removed the part where data by sample are collected, which doesn't have our interest for now.)
#First move all data collected by HybPhaser R script 1a to another directory.
mkdir -p output_HybPhaser_allGenes/01_data_backup/
mv output_HybPhaser_allGenes/01_data/* output_HybPhaser_allGenes/01_data_backup/
#Then move back only the data directorys we need from the namelist, or otherwise the following HybPhaser R scripts will collect sequences from all samples in the directory.
for directory in $(cat ../namelist.txt); do mv output_HybPhaser_allGenes/01_data_backup/${directory} output_HybPhaser_allGenes/01_data/; done
Rscript ../hybphaser_scripts_updated_for_BrassiWood_project/1c_generate_sequence_lists_generalStart.R config_inclusive.txt

#Update the format of the genelist created in 1c_generate_sequence_lists_generalStart, for use in the parallel script below.
awk -F"," '{print $2}' genelist_allGenes_inclusive.txt > genelist_allGenes_inclusive_temp1.txt
sed 's/"x"//g' genelist_allGenes_inclusive_temp1.txt > genelist_allGenes_inclusive_temp2.txt
sed 's/"//g' genelist_allGenes_inclusive_temp2.txt > genelist_allGenes_inclusive_temp3.txt
sed '/^$/d' genelist_allGenes_inclusive_temp3.txt > genelist_allGenes_inclusive.txt
rm genelist_allGenes_inclusive_temp*.txt

parallel -j 67 Rscript ../hybphaser_scripts_updated_for_BrassiWood_project/1c_generate_sequence_lists_perLocus.R config_inclusive.txt :::: genelist_allGenes_inclusive.txt

#Create a backup of the sequence lists just created before pruning the sequence lists based on paralogs etc.
#These sequence lists can equally well be used for pruning for other routines!
#Remove any previous versions.
rm -r output_HybPhaser_allGenes/03_sequence_lists_backup/
#Create a new directory to store sequence lists.
mkdir -p output_HybPhaser_allGenes/03_sequence_lists_backup/
#Now copy latest results to the new directory.
cp -r output_HybPhaser_allGenes/03_sequence_lists_inclusive/* output_HybPhaser_allGenes/03_sequence_lists_backup/

#Then run the final part of script 1c, which is general for all loci and removes loci (and samples from loci) as based on paralog calls from script 1b.
#Rscript ../hybphaser_scripts_updated_for_BrassiWood_project/1c_generate_sequence_lists_generalEnd.R config_inclusive.txt

#Move up one directory.
cd ..
date
