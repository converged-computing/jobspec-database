#!/bin/bash nextflow


/* 
 * enables modules 
 */
nextflow.enable.dsl = 2

// import modules
include { Partition } from './modules/Partition'
include { Combine } from './modules/Combine'

def helpmessage() {

log.info"""

HASE meta-analyzer ~ v${workflow.manifest.version}"
=================================================
Pipeline for running encoded meta-analysis over multiple studies in the central site.

Pay attention that it does not explicitly overwrite the output folder, so clean it before re-running the pipeline.

""".stripIndent()

}

//Default parameters
params.input = ''
params.outdir = ''

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
summary['Input path']                               = params.input
summary['Output directory']                         = params.outdir

log.info summary.collect { k,v -> "${k.padRight(21)}: $v" }.join("\n")
log.info "================================================="


// Process input file paths

parquet = Channel.fromPath(params.input, glob: true)
    .ifEmpty { error "Cannot find input: ${params.input}" }
    .collect()

out = Channel.fromPath(params.outdir)

workflow {

phenotypes = channel.from("ENSG00000004487", "ENSG00000010626", "ENSG00000028839", "ENSG00000059758", "ENSG00000180481")
Partition(parquet)
Combine(Partition.out, phenotypes, out)

}

workflow.onComplete {
    println ( workflow.success ? "Pipeline finished!" : "Something crashed...debug!" )
}
