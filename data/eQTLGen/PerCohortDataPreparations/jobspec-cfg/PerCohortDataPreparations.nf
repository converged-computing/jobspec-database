#!/usr/bin/env nextflow
def helpmessage() {

log.info"""
PerCohortDataPreparations
==============================================
Pipeline for running data preparation and encoding steps in individual datasets, in order to enable centralized meta analysis over all datasets.

This pipeline mostly recapitulates the steps as outlined in HASE wiki page (https://github.com/roshchupkin/hase/wiki/HD-Meta-analysis-GWAS), with some bugfixes.

Usage:

nextflow run PerCohortDataPreparations.nf \
--hdf5 [Genotype hdf5 folder] \
--qcdata [Path to QCd data] \
--outdir [Output folder]

Mandatory arguments:
--hdf5            Path to input genotype folder. It has to be in hdf5 format and contain at least 3 subfolders (genotypes, probes, individuals).
--qcdata          Path to quality-controlled data where expression and covariate files are. This is the output of DataQc pipeline.
--outdir          Path to output folder where encoded, prepared and organised data is written.


""".stripIndent()

}

// Default parameters
params.hdf5 = ''
params.qcdata = ''
params.outputpath = ''
params.permute = 'ues'

log.info """================================================================
HASE per-cohort preparation pipeline v${workflow.manifest.version}"
================================================================"""
def summary = [:]
summary['Pipeline Version']                         = workflow.manifest.version
summary['Current user']                             = "$USER"
summary['Current home']                             = "$HOME"
summary['Current path']                             = "$PWD"
summary['Working dir']                              = workflow.workDir
summary['Script dir']                               = workflow.projectDir
summary['Config Profile']                           = workflow.profile
summary['Container Engine']                         = workflow.containerEngine
if(workflow.containerEngine) summary['Container']   = workflow.container
summary['Input genotype directory']                 = params.hdf5
summary['Input folder QCd data']                    = params.qcdata
summary['Output directory']                         = params.outdir

log.info summary.collect { k,v -> "${k.padRight(21)}: $v" }.join("\n")
log.info "======================================================="

Genotypes = Channel.fromPath(params.hdf5)
.ifEmpty { exit 1, "Genotype folder not found: ${params.hdf5}" }

Genotypes.into{genotypes_to_cohortname; genotypes_to_mapper; genotypes_to_encoding; genotypes_to_pd; genotypes_to_perm_encoding; genotypes_to_perm_pd; genotypes_to_genpc; 
genotypes_to_organise_data}

snpqc = Channel.fromPath(params.hdf5 + '/SNPQC/')
.ifEmpty { exit 1, "SNPQC file not found!" }

snp_probes = Channel.fromPath(params.hdf5 + '/probes/')
.ifEmpty { exit 1, "SNP probes file not found!" }

expression = Channel.fromPath(params.qcdata + '/outputfolder_exp/exp_data_QCd/exp_data_preprocessed.txt')
.ifEmpty { exit 1, "Expression data not found!" }
.into{expression_to_encoding; expression_to_pd; expression_to_permutation}

covariates = Channel.fromPath(params.qcdata + '/CovariatePCs.txt')
.ifEmpty { exit 1, "Covariate data not found!" }
.into{covariates_to_pd; covariates_to_permutation; covariates_to_genpc}

qcdata = Channel.fromPath(params.qcdata)
.ifEmpty { exit 1, "Report etc. path not found!" }

// Parse study name

process ParseCohortName {

  input:
    path geno from genotypes_to_cohortname

  output:
    env cohortname into cohortname

  """
  cohortname=\$(ls ${geno}/individuals/*h5 | sed -e 's/.h5//g' | sed -e 's/.*\\///g')
  echo \$cohortname
  """

}

studyname = cohortname.first().into{studyname_mapper; studyname_encode; studyname_pd; studyname_encode_permuted; studyname_pd_permuted; 
studyname_GenRegPcs; studyname_OrganizeEncodedData; studyname_ReplaceSampleNames}

process CreateMapperFiles {

    input:
      path genopath from genotypes_to_mapper
      val studyname from studyname_mapper

    output:
      path './mapper/' into mapper_to_encode, mapper_to_pd, mapper_to_perm_encode, mapper_to_perm_pd, mapper_to_organize

    """
    python2 $baseDir/bin/hase/tools/mapper.py \
    -g ${genopath} \
    -study_name $studyname \
    -o ./mapper/ \
    -chunk 35000000 \
    -ref_name 1000G-30x_ref \
    -ref_chunk 1000000 \
    -probe_chunk 1000000
    """
}

process EncodeData {

    input:
      path mapper from mapper_to_encode
      path genopath from genotypes_to_encoding
      path expression from expression_to_encoding
      val studyname from studyname_encode

    output:
      file './encoded/' into encoded

    """
    echo ${genopath}
    echo ${mapper}
    echo ${expression}

    mkdir -p input_expression
    mv ${expression} input_expression/.

    python2 $baseDir/bin/hase/hase.py \
    -g ${genopath} \
    -study_name ${studyname} \
    -o ./encoded/ \
    -mapper ${mapper}/ \
    -ref_name 1000G-30x_ref \
    -ph input_expression \
    -mode encoding

    # Remove random matrices to make back-encoding impossible
    rm ./encoded/F*
    """
}

process PartialDerivatives {

    input:
      path mapper from mapper_to_pd
      path genopath from genotypes_to_pd
      path expression from expression_to_pd
      path covariates from covariates_to_pd
      val studyname from studyname_pd

    output:
      file './pd/' into pd

    """
    mkdir -p input_expression
    mkdir -p input_covariates

    mv ${expression} input_expression/.
    mv ${covariates} input_covariates/.
    
    python2 $baseDir/bin/hase/hase.py \
    -g ${genopath}/ \
    -study_name ${studyname} \
    -ph input_expression \
    -cov input_covariates \
    -mapper ${mapper}/ \
    -ref_name 1000G-30x_ref \
    -o ./pd/ \
    -mode single-meta
    """
}

process PermuteData {

  input:
    path expression from expression_to_permutation
    path covariates from covariates_to_permutation

  when:
    params.permute == "ues"

  output:
    path 'shuffled_expression_folder' into perm_exp_to_encoding, perm_exp_to_pd
    path 'shuffled_covariates_folder' into perm_cov_to_encoding, perm_cov_to_pd

  """
  mkdir -p shuffled_expression_folder
  mkdir -p shuffled_covariates_folder

  Rscript --vanilla $baseDir/bin/helperscripts/shuffle_sample_ids.R \
  ${expression} \
  ${covariates} \
  ./shuffled_expression_folder/shuffled_expression.txt \
  ./shuffled_covariates_folder/shuffled_covariates.txt
  """
}

process EncodeDataPermuted {

    input:
      path mapper from mapper_to_perm_encode
      path genopath from genotypes_to_perm_encoding
      path expression from perm_exp_to_encoding
      val studyname from studyname_encode_permuted

    output:
      file './encoded_permuted/' into encoded_permuted

    """
    python2 $baseDir/bin/hase/hase.py \
    -g ${genopath} \
    -study_name ${studyname} \
    -o ./encoded/ \
    -mapper ${mapper}/ \
    -ref_name 1000G-30x_ref \
    -ph ${expression} \
    -mode encoding

    # Remove random matrices to make back-encoding impossible
    rm ./encoded/F*

    mv encoded encoded_permuted
    """
}


process PartialDerivativesPermuted {

    input:
      path mapper from mapper_to_perm_pd
      path genopath from genotypes_to_perm_pd
      path expression from perm_exp_to_pd
      path covariates from perm_cov_to_pd
      val studyname from studyname_pd_permuted

    output:
      file './pd_permuted/' into pd_permuted

    """
    python2 $baseDir/bin/hase/hase.py \
    -g ${genopath}/ \
    -study_name ${studyname} \
    -ph ${expression}/ \
    -cov ${covariates}/ \
    -mapper ${mapper}/ \
    -ref_name 1000G-30x_ref \
    -o ./pd/ \
    -mode single-meta

    mv pd pd_permuted
    """
}

process PrepareGenRegPcs {

    input:
      path covariates from covariates_to_genpc

    output:
      path cov_folder
      path pheno_folder

    """
    # Split covariate file to two pieces: covariates (10 first MDS) and 100 first PCs
    awk -F'\t' '{ print \$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10, \$11}' ${covariates} > covariate_MDS.txt
    awk 'BEGIN{FS=OFS="\t"}{printf \$1"\t"}{for(i=12;i<=NF-1;i++) printf \$i"\t"}{print \$NF}' ${covariates} > pheno_expPC.txt

    mkdir -p cov_folder
    mkdir -p pheno_folder

    mv covariate_MDS.txt cov_folder/.
    mv pheno_expPC.txt pheno_folder/.
    """
}

process RunGenRegPcs {

    tag{"Chunk: $Chunk"}

    input:
      path genopath from genotypes_to_genpc
      path covariates from cov_folder
      path phenotypes from pheno_folder
      val studyname from studyname_GenRegPcs
      each Chunk from 1..100

    output:
      file '*_GenRegPcs.txt' into genetic_pcs

    """
    [ ! -d ${studyname} ] && [ ! -L ${studyname} ] && mv ${genopath} ${studyname}

    python2 $baseDir/bin/hase/hase.py \
    -g ${studyname} \
    -study_name ${studyname} \
    -o output \
    -ph pheno_folder \
    -cov cov_folder \
    -th 3 \
    -mode regression \
    -maf 0.01 \
    -node 100 ${Chunk} \
    -cluster "y"

    # calculate degrees of freedom
    N=\$(wc -l pheno_folder/pheno_expPC.txt | awk '{print \$1}')
    N=\$((N-1))
    # nr. of covariates (first column is sample ID but one needs to add SNP here as well)
    # So this is correct (10 gen PCs + SNP = 11)
    N_cov=\$(awk -F' ' '{print NF; exit}' cov_folder/covariate_MDS.txt)
    df=\$((N - N_cov - 1))

    python2 $baseDir/bin/helperscripts/HaseOutputNumpyAnalyzer.py \
    -i "output/node${Chunk}_*.npy" \
    -df \${df} \
    -o ${Chunk}_GenRegPcs_temp.txt \
    -sref $baseDir/bin/hase/data/1000G-30x.ref.gz

    # Filter in only 1e-5
    awk '{if(NR == 1) {print \$0} else {if(\$9 < 1e-5) { print }}}' ${Chunk}_GenRegPcs_temp.txt > ${Chunk}_GenRegPcs.txt
    rm ${Chunk}_GenRegPcs_temp.txt
    """
}

process OrganizeEncodedData {

    input:
      path pd from pd
      path encoded from encoded
      path pd_permuted from pd_permuted
      path encoded_permuted from encoded_permuted
      path mapper from mapper_to_organize
      path genopath from genotypes_to_organise_data
      path snp_probes from snp_probes
      path snpqc from snpqc
      path qc_data, stageAs: 'output2' from qcdata
      val studyname from studyname_OrganizeEncodedData
      file genetic_pcs from genetic_pcs.collectFile(name: 'GenRegPcs.txt', keepHeader: true, sort: true)

    output:
      path '*' into OrganizedFiles

    """
    # empirical
    mkdir -p ${studyname}_IntermediateFilesEncoded_to_upload/empirical/EncodedGenotypeData/genotype
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/empirical/EncodedGenotypeData/individuals
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/empirical/EncodedPhenotypeData
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/empirical/pd_shared

    cp -r ${snp_probes} ${studyname}_IntermediateFilesEncoded_to_upload/empirical/EncodedGenotypeData/
   
    cp ./${encoded}/encode_individuals/* ${studyname}_IntermediateFilesEncoded_to_upload/empirical/EncodedGenotypeData/individuals/
    cp ./${encoded}/encode_genotype/* ${studyname}_IntermediateFilesEncoded_to_upload/empirical/EncodedGenotypeData/genotype/
    cp ./${encoded}/encode_phenotype/* ${studyname}_IntermediateFilesEncoded_to_upload/empirical/EncodedPhenotypeData/
    
    cp ./${pd}/*.npy ${studyname}_IntermediateFilesEncoded_to_upload/empirical/pd_shared/

    # permuted
    mkdir -p ${studyname}_IntermediateFilesEncoded_to_upload/permuted/EncodedGenotypeData/genotype
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/permuted/EncodedGenotypeData/individuals
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/permuted/EncodedPhenotypeData
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/permuted/pd_shared

    cp ./${encoded_permuted}/encode_individuals/* ${studyname}_IntermediateFilesEncoded_to_upload/permuted/EncodedGenotypeData/individuals/
    cp ./${encoded_permuted}/encode_genotype/* ${studyname}_IntermediateFilesEncoded_to_upload/permuted/EncodedGenotypeData/genotype/
    cp ./${encoded_permuted}/encode_phenotype/* ${studyname}_IntermediateFilesEncoded_to_upload/permuted/EncodedPhenotypeData/
    
    cp ./${pd_permuted}/*.npy ${studyname}_IntermediateFilesEncoded_to_upload/permuted/pd_shared/

    # Additional files needed for diagnostics
    cp -r ./${mapper} ${studyname}_IntermediateFilesEncoded_to_upload/
    
    # Summary files and plots from QCd data folder
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/SumStats
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/expression/
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/expression/plots
    # Gene expression summary statistics (mean, median, sd, etc.)
    cp -r ${qc_data}/outputfolder_exp/exp_data_summary/raw_gene_summary.txt ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/expression/.
    cp -r ${qc_data}/outputfolder_exp/exp_data_summary/processed_gene_summary.txt ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/expression/.
    # plots from QC report
    cp -r ${qc_data}/outputfolder_exp/exp_plots/*pdf ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/expression/plots/.

    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/genotypes/
    mkdir ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/genotypes/plots
    # genotype QC summary
    cp -r ${snpqc} ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/genotypes/.
    # plots from QC report
    cp -r ${qc_data}/outputfolder_gen/gen_plots/*pdf ${studyname}_IntermediateFilesEncoded_to_upload/SumStats/genotypes/plots/.
  
    # Data QC report
    cp -r ${qc_data}/Report_DataQc* ${studyname}_IntermediateFilesEncoded_to_upload/.
    
    # Genetically regulated PCs
    mv GenRegPcs.txt ${studyname}_GenRegPcs.txt
    gzip -f ${studyname}_GenRegPcs.txt
    cp ${studyname}_GenRegPcs.txt.gz ${studyname}_IntermediateFilesEncoded_to_upload/.
    """
}

process ReplaceSampleNames {

    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
      path OrganizedFiles from OrganizedFiles
      val studyname from studyname_ReplaceSampleNames

    output:
      path "${studyname}_IntermediateFilesEncoded_to_upload" into IntermediateFilesEncodedSampleIdsReplaced_to_upload
      file "*.md5" into md5sumfile

    """
    python2 $baseDir/bin/helperscripts/replace_sample_names.py -IntFileEnc ${studyname}_IntermediateFilesEncoded_to_upload/empirical/
    python2 $baseDir/bin/helperscripts/replace_sample_names.py -IntFileEnc ${studyname}_IntermediateFilesEncoded_to_upload/permuted/

    find ${studyname}_IntermediateFilesEncoded_to_upload/ -type f -print0 | xargs -0 md5sum > ${studyname}_OrganizedFiles.md5
    """
}

workflow.onComplete {
    println ( workflow.success ? "Pipeline finished!" : "Something crashed...debug!" )
}
