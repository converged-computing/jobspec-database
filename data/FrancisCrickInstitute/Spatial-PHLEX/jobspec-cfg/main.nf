#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*********************************************
* RUBICON NEXTFLOW SPATIAL ANALYSIS PIPELINE *
*********************************************/

include {NEIGHBOURHOOD_WF; NEAREST_NEIGHBOUR_WF} from './workflows/barrier.nf'
include {SPATIAL_CLUSTERING_WF; CLUSTERED_BARRIER_WF; NEIGHBOURS_WF } from './workflows/spatial.nf'
include { print_logo; check_params } from './modules/util.nf'

project_dir = projectDir

println projectDir
println workDir
println "$workDir/tmp"

// directly create a list channel for phenotyping levels for combinatoric input of phenotype levels and imagenames
pheno_list = params.phenotyping_column?.tokenize(',')
ch_phenotyping = Channel.fromList(pheno_list)

// channel for neighbourhood input csv files
if (params.neighborhood_input) {
    Channel
        .fromPath(params.neighborhood_input, checkIfExists: true)
        .take( params.dev ? params.number_of_inputs : -1 )
        .map { it }
        .ifEmpty { exit 1, "Input file not found: ${params.neighborhood_input}" }
        .set { ch_nhood }
} 

ch_phenotyping = ch_phenotyping.first() // this will take only the first phenotyping level even if multiple are specified -- > deprecate multiple input?

workflow {
    
    print_logo()
    check_params()
    
    if ((params.workflow_name == 'default') || (params.workflow_name == 'clustered_barrier')) {
        CLUSTERED_BARRIER_WF ( params.objects, ch_phenotyping)
    } else if (params.workflow_name == 'spatial_clustering') {
        SPATIAL_CLUSTERING_WF ( params.objects, ch_phenotyping)
    } else if (params.workflow_name == 'barrier_only') {

        if (params.graph_type == 'neighbouRhood') {
            NEIGHBOURHOOD_WF ( ch_nhood, params.neighbourhood_module_no, params.objects, params.objects_delimiter)
        }
        if (params.graph_type == 'nearest_neighbour') {
            NEAREST_NEIGHBOUR_WF (params.objects)
        }
    } else if (params.workflow_name == 'neighbours') {
        NEIGHBOURS_WF(params.objects)
    } else {
        println "Workflow name not recognised. Please choose from: 'default', 'clustered_barrier', 'spatial_clustering', 'barrier_only', 'neighbours'"
    }

}