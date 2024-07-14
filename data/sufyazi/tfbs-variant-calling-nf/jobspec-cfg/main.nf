#!/usr/bin/env nextflow

// All of the default parameters are being set in `nextflow.config`
// Import sub-workflows
include { extractTFBSBeds } from './modules/extractTFBSBeds'
include { generateBAMPaths } from './modules/generateBAMPaths'
include { callVariants } from './modules/callVariants'

// Function which prints help message text
def helpMessage() {
    log.info"""
Usage:

nextflow run main.nf <--OPTION NAME> <ARGUMENT>

Required Arguments:
    --run_mode                    Set the run mode for the workflow. Options are `all` for all available dataset IDs in the base directory [--bam_dir] or `subset` for just the IDs specified in the list provided with [--dataset_id_list]. Default is `all` if not explicitly set
    --help                        Print this help message and exit

  Input paths:
    --bam_dir                     Path to base directory where directories of datasets containing BAM files and their respective bai index files are located [MANDATORY]
    --genome_fasta                Path to the reference genome fasta file [MANDATORY]
    --fpscore_matrix              Globbed path to base directory (path/to/dir/*.ext) where the merged footprint score matrix files of all studied motifs are located (saved as parquet files) [MANDATORY]

  Input manifest:
    --dataset_id_list             Path to a list of dataset IDs to work on. Required if [--run_mode] is set to `subset`

  Output path:
    --output_dir                  Directory path for output VCF files [MANDATORY]
    """.stripIndent()
}


// Main workflow
workflow {
    // Show help message if the user specifies the --help flag at runtime
    // or if any required params are not provided
    if ( params.help || params.bam_dir == false || params.output_dir == false || params.genome_fasta == false || params.fpscore_matrix == false ){
        // Invoke the function above which prints the help message
        helpMessage()
        // Exit out and do not run anything else
        exit 1
    } else {
        log.info "Proceeding with workflow..."

        // Preparing the input data
        log.info "Preparing the input data..."

        def matrixFiles = []
        
        log.info "Checking whether the directory of footprint score matrices is valid..."

        // Check if fpscore_matrix is not valid dir or does not exist
        if ( file(params.fpscore_matrix).isDirectory() == false || file(params.fpscore_matrix).exists() == false ) {
            log.error "The footprint matrix directory is not a valid directory or does not exist. Please provide a valid directory path."
            exit 1
        }
        else {
            log.info "The footprint matrix provided is valid. Getting a list of the matrix files..."
            file(params.fpscore_matrix).eachFileMatch(~/.*\.parquet$/) { file ->
                                                                        matrixFiles << file
                                                                        }
            log.info "Total number of matrix files found: ${matrixFiles.size}"
            if (matrixFiles.size() == 0){
                log.error "No matrix files were found in the provided directory. Please provide a valid directory path containing matrix files with the .parquet extension."
                exit 1
            } else {
                //log.info "Printing matrix file no. 1: ${matrixFiles[0]}"
                log.info "Execution of workflow will proceed..."
            }
        }
    
        // Set up a channel to grab all the matrix files in the input folder
        in_matrix = Channel.fromPath(matrixFiles)//.view()
        // Extract the prefix from the input files and return a tuple of the file and the prefix
        motifMatrix_ch = in_matrix.map{ file -> [file.baseName.replaceAll("_tfbs_merged_matrix-full", ""), file] }//.view()
        // Extract the TF footprint regions (TFBS) from the input fps matrix files as bed files
        bedFilesList = extractTFBSBeds(motifMatrix_ch).toList()//.view()


        // Check if run_mode is set
        if (params.run_mode == "subset"){

            // Check if dataset_id_list is provided
            if (params.dataset_id_list == false){
                log.error "The [--dataset_id_list] option is required to run the workflow if [--run_mode] is set to 'subset'."
                exit 1
            } else {

                // Extract bam directory paths
                datasetIDs_ch = Channel.fromPath(params.dataset_id_list).splitText().map { it.trim() }//.view()
                datasetIDPaths_ch = datasetIDs_ch.map { id -> [id, file("${params.bam_dir}/${id}")] }//.view()
            }
        }
        else if ( params.run_mode == "all" ) {

            // Extract all unique dataset IDs in the input bam folder and the path to the dataset ID
            datasetIDPaths_ch = Channel.fromPath("${params.bam_dir}/*", type: 'dir').map { dir -> [dir.name, dir] }//.view()

        }

        // Set up a channel to grab all the bam files for each dataset ID
        datasetIDBams_ch = datasetIDPaths_ch.map { id, path -> 
                                            def bams = files("${path}/**.bam")
                                            return [id, bams]
                                            }//.view()

        // Generate a channel for the bam list of a dataset ID
        bamPaths_ch = generateBAMPaths(datasetIDBams_ch)//.view()

        // now we can run the variant-calling process
        rawVCFs_ch = callVariants(bamPaths_ch, bedFilesList)//.view()

    }
}
workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
}