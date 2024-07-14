#!/bin/bash

#This file contains the true analyses script per samples.

sample=$1

echo Merging HybPhaser output for sample ${sample};

#Remove any directory for the sample if it already exists, to prevent data from different runs being merged unnoticed.
rm -r results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}

#Create a sample-specific output directory.
mkdir -p results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}

#Create a directory for the reads and combine.
mkdir -p results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/reads/
cp results_HybPhaser_Nikolov2019/output_HybPhaser_Nikolov2019/01_data/${sample}/reads/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/reads/
cp results_HybPhaser_Angiosperms353NewTargets/output_HybPhaser_Angiosperms353NewTargets/01_data/${sample}/reads/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/reads/

#Create a directory for the mapping_files and combine.
mkdir -p results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/mapping_files/
cp results_HybPhaser_Nikolov2019/output_HybPhaser_Nikolov2019/01_data/${sample}/mapping_files/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/mapping_files/
cp results_HybPhaser_Angiosperms353NewTargets/output_HybPhaser_Angiosperms353NewTargets/01_data/${sample}/mapping_files/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/mapping_files/

#Create a directory for the contigs and combine.
mkdir -p results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/contigs/
cp results_HybPhaser_Nikolov2019/output_HybPhaser_Nikolov2019/01_data/${sample}/contigs/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/contigs/
cp results_HybPhaser_Angiosperms353NewTargets/output_HybPhaser_Angiosperms353NewTargets/01_data/${sample}/contigs/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/contigs/

#Create a directory for the consensus sequences and combine.
mkdir -p results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/consensus/
cp results_HybPhaser_Nikolov2019/output_HybPhaser_Nikolov2019/01_data/${sample}/consensus/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/consensus/
cp results_HybPhaser_Angiosperms353NewTargets/output_HybPhaser_Angiosperms353NewTargets/01_data/${sample}/consensus/* results_HybPhaser_allGenes/output_HybPhaser_allGenes/01_data/${sample}/consensus/
