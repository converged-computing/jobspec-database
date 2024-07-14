#!/usr/bin/env nextflow

// Define DSL2
nextflow.enable.dsl=2

// Don't overwrite global params.modules, create a copy instead and use that within the main script.
def modules = params.modules.clone()

def skip_seurat_filtering = params.skip_seurat_filtering ? true : false
// def skip_scvelo = params.skip_scvelo ? true : false

/*-----------------------------------------------------------------------------------------------------------------------------
Log
-------------------------------------------------------------------------------------------------------------------------------*/
if(params.debug) {log.info Headers.build_debug_param_summary(params, params.monochrome_logs)}

def analysis_scripts                                = [:]
analysis_scripts.transfer_labels                    = file("$baseDir/bin/seurat/transfer_labels.R", checkIfExists: true)
analysis_scripts.gene_modules_subset_latent_time    = file("$baseDir/bin/other/gene_modules_subset_latent_time.R", checkIfExists: true)
analysis_scripts.gene_modules_npb_latent_time       = file("$baseDir/bin/other/gene_modules_npb_latent_time.R", checkIfExists: true)
analysis_scripts.coexpression_analysis_npb          = file("$baseDir/bin/other/coexpression_analysis_npb.R", checkIfExists: true)
analysis_scripts.coexpression_nc_ppr_modules_npb    = file("$baseDir/bin/other/coexpression_nc_ppr_modules_npb.R", checkIfExists: true)
analysis_scripts.split                              = file("$baseDir/bin/seurat/split_seurat.R", checkIfExists: true)
analysis_scripts.cluster                            = file("$baseDir/bin/seurat/subset_cluster.R", checkIfExists: true)

/*------------------------------------------------------------------------------------*/
/* Workflow inclusions
--------------------------------------------------------------------------------------*/

include {METADATA} from "$baseDir/subworkflows/metadata/main"

include {SEURAT_FILTERING} from "$baseDir/subworkflows/seurat_filtering/main"                                                           addParams(  preprocessing_options:                  modules['preprocessing'],
                                                                                                                                                    integration_options:                    modules['integration'],
                                                                                                                                                    integration_qc_options:                 modules['integration_qc'],
                                                                                                                                                    sex_filt_options:                       modules['sex_filt'],
                                                                                                                                                    cell_cycle_options:                     modules['cell_cycle'],
                                                                                                                                                    contamination_filt_options:             modules['contamination_filt'])

// Modules and subworkflows for running scVelo/cellrank                                             
include {MERGE_LOOM} from "$baseDir/modules/local/merge_loom/main"                                                                      addParams(  options:                                modules['merge_loom'])

include {SEURAT_FILTERED_PROCESS} from "$baseDir/subworkflows/seurat_filtered_process/main"                                             addParams(  gene_module_options:                    modules['gene_modules'],
                                                                                                                                                    state_classification_options:           modules['state_classification'],
                                                                                                                                                    seurat_h5ad_options:                    modules['seurat_h5ad'],
                                                                                                                                                    seurat_intersect_loom_options:          modules['seurat_intersect_loom'],
                                                                                                                                                    scvelo_run_options:                     modules['scvelo_run'],
                                                                                                                                                    cellrank_run_options:                   modules['cellrank_run'])

// Subworkflows for split stage and run and run downstream analysis
include {SEURAT_SPLIT_PROCESS as SEURAT_STAGE_PROCESS} from "$baseDir/subworkflows/seurat_split_process/main"                           addParams(  split_options:                          modules['stage_split'],
                                                                                                                                                    cluster_options:                        modules['stage_cluster'],
                                                                                                                                                    gene_modules_options:                   modules['stage_gene_modules'],
                                                                                                                                                    state_classification_options:           modules['stage_state_classification'],
                                                                                                                                                    seurat_h5ad_options:                    modules['seurat_h5ad'],
                                                                                                                                                    seurat_intersect_loom_options:          modules['stage_seurat_intersect_loom'],
                                                                                                                                                    scvelo_run_options:                     modules['stage_scvelo_run'])


// Subworkflows for label transfer and subsequent cluster subsets and run downstream analysis

include {R as TRANSFER_LABELS} from "$baseDir/modules/local/r/main"                                                                     addParams(  options:                                modules['transfer_labels'],
                                                                                                                                                    script:                                 analysis_scripts.transfer_labels )

// Subworkflow to create new schelper labels on data with contamination
include {SEURAT_SPLIT_PROCESS as SEURAT_STAGE_PROCESS_CONTAM} from "$baseDir/subworkflows/seurat_split_process/main"                           addParams(  split_options:                          modules['stage_split'],
                                                                                                                                                    cluster_options:                        modules['stage_cluster_contam'],
                                                                                                                                                    gene_modules_options:                   modules['stage_gene_modules'],
                                                                                                                                                    state_classification_options:           modules['stage_state_classification_contam'],
                                                                                                                                                    seurat_h5ad_options:                    modules['seurat_h5ad'],
                                                                                                                                                    seurat_intersect_loom_options:          modules['stage_seurat_intersect_loom'],
                                                                                                                                                    scvelo_run_options:                     modules['stage_scvelo_run'])

// Subworkflow to transfer labels for integration with scATAC
include {INTEGRATION_PREP} from "$baseDir/subworkflows/seurat_integration_prep/main"                           addParams(  contamination_ident_options:             modules['contamination_ident'],
                                                                                                                           transfer_labels_options:                 modules['transfer_labels_for_integration'],
                                                                                                                           subset_options:                          modules['subset_remove_HH4'],
                                                                                                                           cluster_options:                         modules['HH4_subset_cluster'],)

include {R as TRANSFER_LABELS_INTEGRATION} from "$baseDir/modules/local/r/main"                                                                     addParams(  options:                    modules['transfer_labels_integration'],
                                                                                                                                                    script:                                 analysis_scripts.transfer_labels )
include {R as SPLIT_STAGE_INTEGRATION} from "$baseDir/modules/local/r/main"                     addParams(  options: modules['split_stage_integration'],
                                                                                                script: analysis_scripts.split )
include {R as CLUSTER_STAGE_INTEGRATION} from "$baseDir/modules/local/r/main"                   addParams(  options: modules['cluster_stage_integration'],
                                                                                                script: analysis_scripts.cluster )



// Set channel for binary knowledge matrix for cell state classification
Channel
    .value("$baseDir/binary_knowledge_matrix.csv")
    .set{ch_binary_knowledge_matrix}

Channel
    .value("$baseDir/binary_knowledge_matrix_contam.csv")
    .set{ch_binary_knowledge_matrix_contam}

workflow {
    METADATA( params.input )

    /*------------------------------------------------------------------------------------*/
    /* Run inital seurat pipeline
    --------------------------------------------------------------------------------------*/
    // Set channel for cellranger counts
    METADATA.out
        .filter{ it[0].sample_id == 'NF-scRNAseq_alignment_out' }
        .map {[it[0], it[1].collect{ file(it+"/cellranger/count/filtered_feature_bc_matrix", checkIfExists: true) }]}
        .set {ch_scRNAseq_counts}

    SEURAT_FILTERING( ch_scRNAseq_counts )
        
    /*------------------------------------------------------------------------------------*/
    /* Prepare inputs for scVelo
    --------------------------------------------------------------------------------------*/
   
    // Set channel for input looms
    METADATA.out
        .filter{ it[0].sample_id == 'NF-scRNAseq_alignment_out' }
        .map {[it[0], it[1].collect{ file(it+"/velocyto", checkIfExists: true) }]}
        .set {ch_loomInput}

    MERGE_LOOM( ch_loomInput )

    /*------------------------------------------------------------------------------------*/
    /* Run analysis on stage and run split
    --------------------------------------------------------------------------------------*/ 
    SEURAT_STAGE_PROCESS( SEURAT_FILTERING.out.contamination_filt_out, MERGE_LOOM.out.loom.map{it[1]}, SEURAT_FILTERING.out.annotations.map{it[1]}, ch_binary_knowledge_matrix )

    /*------------------------------------------------------------------------------------*/
    /* Transfer cell type labels from stage to full dataset
    --------------------------------------------------------------------------------------*/     

    // Collect rds files from all stages
    ch_combined = SEURAT_STAGE_PROCESS.out.state_classification_out
        .concat(SEURAT_FILTERING.out.contamination_filt_out)
        .map{it[1].findAll{it =~ /rds_files/}[0].listFiles()[0]}
        .collect()
        .map { [[sample_id:'all_stages_filtered'], it] } // [[meta], [rds1, rds2, rds3, ...]]

    // Transfer labels from stage subsets to full data
    TRANSFER_LABELS( ch_combined )

    /*------------------------------------------------------------------------------------*/
    /* Prepare data for integration with ATAC
    --------------------------------------------------------------------------------------*/  

    // Re-run classification but including contaminating populations - in scHelper_cell_type slot
    SEURAT_STAGE_PROCESS_CONTAM( SEURAT_FILTERING.out.cell_cycle_out, MERGE_LOOM.out.loom.map{it[1]}, SEURAT_FILTERING.out.annotations.map{it[1]}, ch_binary_knowledge_matrix_contam )
    
    // Extract original scHelper cell type labels and add to data - in scHelper_cell_type slot
    INTEGRATION_PREP( SEURAT_FILTERING.out.cell_cycle_out, TRANSFER_LABELS.out )
    // INTEGRATION_PREP.out
    //     .map {row -> [row[0], row[1].findAll { it =~ ".*rds_files" }]}
    //     .flatMap {it[1][0].listFiles()}
    //     .map { [[sample_id:'old_labels'], it] }
    //     .set { ch_old_labels }   

    ch_labels = SEURAT_STAGE_PROCESS_CONTAM.out.state_classification_out
        .concat(INTEGRATION_PREP.out.cluster_out)
        .map{it[1].findAll{it =~ /rds_files/}[0].listFiles()[0]}
        .collect()
        .map { [[sample_id:'all_stages_filtered'], it] } // [[meta], [rds1, rds2, rds3, ...]]

    // Collect rds files from all stages with new labels and the full data with old labels
    // ch_labels = SEURAT_STAGE_PROCESS_CONTAM.out.state_classification_out
    //     .map{it[1].findAll{it =~ /rds_files/}[0].listFiles()[0]}
    //     .collect()
    //     .map { [[sample_id:'all_stages'], it] } // [[meta], [rds1, rds2, rds3, ...]]
    //     .combine( ch_old_labels ) //[[sample_id:all_stages], [HH7, ss8, HH6, ss4, HH4, HH5], [sample_id:NF-scRNA-input], [rds_files, plots]]
    //     .map{[it[0], it[1] + it[3]]} //[[sample_id:all_stages], [HH7, ss8, HH6, ss4, HH4, HH5, rds_files, plots]
    //     //.view() //[[sample_id:all_stages], [HH6, HH4, ss8, ss4, HH7, HH5, cell_cycle_data.RDS]]

    // Transfer new labels on stages to scHelper_cell_type_new slot on old full data
    TRANSFER_LABELS_INTEGRATION( ch_labels )

    // Split data so can integrate stage by stage
    SPLIT_STAGE_INTEGRATION( TRANSFER_LABELS_INTEGRATION.out )
    SPLIT_STAGE_INTEGRATION.out
        .map {row -> [row[0], row[1].findAll { it =~ ".*rds_files" }]}
        .flatMap {it[1][0].listFiles()}
        .map { row -> [[sample_id:row.name.replaceFirst(~/\.[^\.]+$/, '')], row] }
        .set { ch_split_run }  
    CLUSTER_STAGE_INTEGRATION( ch_split_run )

}