#!/bin/bash

#This file contains the true analyses script per samples.

sample=$1

#Remove any previous results from the HybPhaser directory to prevent old and new results being mixed when moving data.
rm -r results_HybPhaser_Angiosperms353NewTargets/${sample}/

#Move HybPiper-mapped data to the HybPhaser mapped data directory.
#mv mapped_Angiosperms353NewTargets/${sample}/ results_HybPhaser_Angiosperms353NewTargets/

#Copy new HybPiper-mapped data to the HybPhaser mapped data directory and unpack.
cp mapped_Angiosperms353NewTargets/${sample}_mapped_Angiosperms353Newtargets.tar.gz results_HybPhaser_Angiosperms353NewTargets/

#Move to that directory.
cd results_HybPhaser_Angiosperms353NewTargets/
tar -xf ${sample}_mapped_Angiosperms353Newtargets.tar.gz


#Reload some modules for this specific parallel run.
module load biocontainers/0.1.0 	#With this module, various other tools/modules can be loaded
module load intel/17.0.4
module load bwa/0.7.16a
module load blast/2.6.0
module load samtools/1.10

#Run step 1 of HybPhaser: generate consensus sequences (exclude data on introns, which includes too much variation in our dataset).
bash /scratch/08079/tg873782/HybPhaser/1_generate_consensus_sequences.sh -s ${sample} -o output_HybPhaser_Angiosperms353NewTargets/

#Move HybPiper-mapped data from the HybPhaser directory back to the original directory.
#mv ${sample}/ ../mapped_Angiosperms353NewTargets/

#Remove the HybPiper data for this sample.
rm -r mapped_Angiosperms353NewTargets/${sample}/
