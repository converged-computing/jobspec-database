#!/bin/bash -l
#PBS -l walltime=336:00:00,mem=5gb,nodes=1:ppn=2
#PBS -o /scratch.global/Salmonella_WGS/WGS_SNP_pipelines/outfile_o_file
#PBS -e /scratch.global/Salmonella_WGS/WGS_SNP_pipelines/errorfile_e_file
#PBS -q max

#PBS -m abe
#PBS -M edoster@umn.edu

module purge
module load singularity

cd /scratch.global/Salmonella_WGS/WGS_SNP_pipelines

nextflow run main_combined_pipeline.nf --reference_genome /scratch.global/Salmonella_WGS/ref_L_monocytogenes_NC_003210.fasta --fastq_dir_path '/scratch.global/Salmonella_WGS/test_GenomeTrakr_L_monocytogenes_WGS_results/Interleaved_fasta/' -profile singularity --output /scratch.global/Salmonella_WGS/test_250_GenomeTrakr_L_monocytogenes_WGS_results --threads 128 -w /scratch.global/Salmonella_WGS/work_250_qsub_l_latest -resume -with-report 250_Listeria_WGS_tools.report -with-trace -with-timeline
