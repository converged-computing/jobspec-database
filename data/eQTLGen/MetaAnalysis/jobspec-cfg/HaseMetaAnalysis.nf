#!/bin/bash nextflow


/* 
 * enables modules 
 */
nextflow.enable.dsl = 2

// import modules
include { MetaAnalyseCohorts } from './modules/MetaAnalyseCohorts'
include { PerCohortAnalysis } from './modules/PerCohortAnalysis'
include { SubsetGenesInclusion } from './modules/SubsetGenesInclusion'

def helpmessage() {

log.info"""

HASE meta-analyzer ~ v${workflow.manifest.version}"
=================================================
Pipeline for running encoded meta-analysis over multiple studies in the central site.

Pay attention that it does not explicitly overwrite the output folder, so clean it before re-running the pipeline.

Usage:

nextflow run HaseMetaAnalysis.nf \
--mastertable [path to master table file] \
--mapperpath [path to mapper files] \
--outdir [path to results folder]

Mandatory arguments:
--mastertable     Path to the master file with following columns. cohort: cohort name, genotype: path to encoded genotype data, expression: path to encoded expression data, partial_derivatives: path to partial derivatives folder, encoded: encoded status 0 or 1, snp_inclusion: path to SNPs inclusion file (header "ID", and SNP IDs on each row) and gene_inclusion: path to gene inclusion file (header "ID", and gene IDs on each row).
--mapperpath      Path to the mappers folder, containing the mapper files for each cohort.
--outdir      Path to results folder where meta-analysis outputs are written.
--covariates      What covariates to include into the analysis for each cohort. If specified, the it has to be the file which has each line formatted as following: [cohort name] [index of covariate to include 1] [index of covariate to include 2] etc. Indexing is 1-based and defaults to all covariates.


Optional arguments:
--th             T-statistic threshold for writing out the results. Defaults to 0.
--genes_percohort File with genes to output cohort specific results for.
--chunks         In how many chunks to run the analysis. Defaults to 100.

""".stripIndent()

}

//Default parameters
params.mastertable = ''
params.genes_percohort = ''
params.outputpath = ''
params.mapperpath = ''
params.th = 0
params.covariates = ''
params.chunks = 100

log.info """=================================================
HASE meta-analyzer v${workflow.manifest.version}"
================================================="""
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
summary['Master table']                             = params.mastertable
summary['Genes per cohort']                         = params.genes_percohort
summary['Input mapper path']                        = params.mapperpath
summary['Output directory']                         = params.outdir
summary['T-statistic threshold']                    = params.th
summary['Covariate filter']                         = params.covariates
summary['Nr. of chunks']                            = params.chunks

log.info summary.collect { k,v -> "${k.padRight(21)}: $v" }.join("\n")
log.info "================================================="


// Process input file paths

cohort_ch = Channel.fromPath(params.mastertable)
    .ifEmpty { error "Cannot find master table from: ${params.mastertable}" }
    .splitCsv(header: true, sep: '\t', strip: true)
    .map{row -> [ row.cohort ]}
    .collect()

genotype_ch = Channel.fromPath(params.mastertable)
    .ifEmpty { error "Cannot find master table from: ${params.mastertable}" }
    .splitCsv(header: true, sep: '\t', strip: true)
    .map{row -> [ row.genotype ]}
    .collect()
    
expression_ch = Channel.fromPath(params.mastertable)
    .ifEmpty { error "Cannot find master table from: ${params.mastertable}" }
    .splitCsv(header: true, sep: '\t', strip: true)
    .map{row -> [ row.expression ]}
    .collect()
    
partial_derivatives_ch = Channel.fromPath(params.mastertable)
    .ifEmpty { error "Cannot find master table from: ${params.mastertable}" }
    .splitCsv(header: true, sep: '\t', strip: true)
    .map{row -> [ row.partial_derivatives ]}
    .collect()
    
encoded_ch = Channel.fromPath(params.mastertable)
    .ifEmpty { error "Cannot find master table from: ${params.mastertable}" }
    .splitCsv(header: true, sep: '\t', strip: true)
    .map{row -> [ "${row.encoded}" ]}
    .collect().view()
    
snp_inclusion_ch = Channel.fromPath(params.mastertable)
    .ifEmpty { error "Cannot find master table from: ${params.mastertable}" }
    .splitCsv(header: true, sep: '\t', strip: true)
    .map{row -> [ "${row.snp_inclusion}" ]}
    .collect()

gene_inclusion_ch = Channel.fromPath(params.mastertable)
    .ifEmpty { error "Cannot find master table from: ${params.mastertable}" }
    .splitCsv(header: true, sep: '\t', strip: true)
    .map{row -> [ "${row.gene_inclusion}" ]}
    .collect()

if (params.genes_percohort) {
  gene_percohort_ch = Channel.fromPath(params.genes_percohort)
   .ifEmpty { error "Cannot find master table from: ${params.genes_percohort}" }
   .collect()
}

mapper = file(params.mapperpath)

test_ch = Channel.fromPath(params.mastertable)

// Optional arguments
th = params.th
chunks = params.chunks

if (params.covariates) {
  covariate_file = file(params.covariates)
} else {covariate_file = Channel.value("")}

chunk = Channel.from(1..chunks)

workflow {
  if (params.genes_percohort) {
    subset_gene_inclusion_ch = SubsetGenesInclusion(gene_inclusion_ch, gene_percohort_ch)
    PerCohortAnalysis(chunk, mapper, th, chunks, snp_inclusion_ch,
      subset_gene_inclusion_ch.collect(), gene_percohort_ch, covariate_file, genotype_ch, expression_ch,
      partial_derivatives_ch, cohort_ch, encoded_ch)
  }
  else {
    MetaAnalyseCohorts(chunk, mapper, th, chunks, snp_inclusion_ch,
      gene_inclusion_ch, covariate_file, genotype_ch, expression_ch,
      partial_derivatives_ch, cohort_ch, encoded_ch)
  }
}

workflow.onComplete {
    println ( workflow.success ? "Pipeline finished!" : "Something crashed...debug!" )
}
