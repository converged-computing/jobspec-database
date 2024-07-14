#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    BORG/vsearch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Github : https://github.com/bio-ontology-research-group/16SProcessing
----------------------------------------------------------------------------------------
*/

params.in_dir = './' // Default input directory
params.out_dir = 'results' // Default output directory
params.skip_cutadapt = false
params.fwd_primer = 'CCTACGGGNGGCWGCAG' // Default forward primer sequence
params.rev_primer = 'GGACTACNVGGGTWTCTAAT' // Default reverse primer sequence
params.db_path = "/data/gold.fasta"
params.rdp_path = "/data/rdp_16s_v18.fa"
params.python_paths = "/data/"
params.single_end = false // false indicates paired-end by default, set to true for single-end

/*
    ================================================================================
                                Preprocessing and QC
    ================================================================================
*/


process CutAdapt {
    label 'preprocess'
    tag "${pair_id}"
    publishDir "${params.out_dir}_results/trimmed_reads", mode: 'copy'

    input:
    tuple val(pair_id), path(reads)

    output:
    tuple val(pair_id), path("*_trimmed.fastq.gz"), emit: trimmed_reads

    script:
    if (params.single_end) {
        def read1 = reads
        """
        cutadapt -g ${params.fwd_primer} -o ${pair_id}_trimmed.fastq.gz ${read1}
        """
    } else {
        def (read1, read2) = reads
        """
        cutadapt -g ${params.fwd_primer} -G ${params.rev_primer} \
            -o ${pair_id}_L001_R1_001_trimmed.fastq.gz \
            -p ${pair_id}_L001_R2_001_trimmed.fastq.gz \
            ${read1} ${read2}
        """
    }
}

process FastQC {
    label 'preprocess'
    tag "${pair_id}"
    publishDir "${params.out_dir}_results/fastQC", mode: 'copy'

    input:
    tuple val(pair_id), path(read_files)

    output:
    path "*.html", emit: fastqc_html
    path "*.zip", emit: fastqc_zip

    script:
    if (params.single_end) {
        def read1 = read_files
        """
        fastqc ${read1}
        """
    } else {
        def (read1, read2) = read_files
        """
        fastqc ${read1} ${read2}
        """
    }
}

/*
    ================================================================================
                                        Merging
    ================================================================================
*/

process VsearchMerge {
    label 'vsearch'
    tag "${pair_id}"
    publishDir "${params.out_dir}_results/intermediate_files", mode: 'copy'

    input:
    tuple val(pair_id), path(trimmed_reads)

    output:
    tuple val(pair_id), path("*.merged.fastq"), emit: merged_fastq
    path "vsearch_merge_log_${pair_id}.txt", emit: merge_log // Added line to emit the log file

    script:
    def (read1, read2) = trimmed_reads
    """
    vsearch --fastq_mergepairs ${read1} --reverse ${read2} --fastqout ${pair_id}.merged.fastq &> vsearch_merge_log_${pair_id}.txt
    """
}


process VsearchStats {
    label 'vsearch'
    tag "${pair_id}"
    publishDir "${params.out_dir}_results/intermediate_files", mode: 'copy'

    input:
    tuple val(pair_id), path(merged_fastq)

    output:
    path "${pair_id}.stats"

    script:
    """
    vsearch --fastq_eestats ${merged_fastq} --output ${pair_id}.stats
    """
}

process VsearchFilter {
    label 'vsearch'
    tag "${pair_id}"
    publishDir "${params.out_dir}_results/intermediate_files", mode: 'copy'

    input:
    tuple val (pair_id), path(merged_fastq)

    output:
    tuple val(pair_id), path("*.filtered.fasta"), emit: filtered_fasta

    script:
    """
    vsearch --fastq_filter ${merged_fastq} --fastq_maxee 1.0 --fastq_minlen 10 --fastq_maxlen 500 --fastq_maxns 0 --fastaout ${pair_id}.filtered.fasta --fasta_width 0
    """
}

process VsearchDereplicate {
    label 'vsearch'
    tag "${pair_id}"
    publishDir "${params.out_dir}_results/intermediate_files", mode: 'copy'

    input:
    tuple val(pair_id), path(filtered_fasta)

    output:
    path "${pair_id}.derep.fasta"

    script:
    """
    vsearch --derep_fulllength ${filtered_fasta} --strand plus --output ${pair_id}.derep.fasta --sizeout --relabel ${pair_id}. --fasta_width 0
    """
}

process MergeAll{
    publishDir "${params.out_dir}_results/merged_intermediates", mode: 'copy'

    input:
    path derep_fasta_files

    output:
    path "all.fasta"

    script:
    """
    cat ${derep_fasta_files.join(' ')} > all.fasta
    """
}

process VsearchDerepAll{
    label 'vsearch'
    publishDir "${params.out_dir}_results/merged_intermediates", mode: 'copy'

    input:
    path all_merged_fasta

    output:
    path "derep.fasta"

    script:
    """
    vsearch --derep_fulllength ${all_merged_fasta} --sizein --sizeout --fasta_width 0 --uc all.derep.uc --output derep.fasta
    """
}

/*
    ================================================================================
                                        Clustering
    ================================================================================
*/

process VsearchCluster{
    label 'vsearch'
    publishDir "${params.out_dir}_results/merged_intermediates", mode: 'copy'

    input:
    path all_derep_fasta

    output:
    path "centroids.fasta"

    script:
    """
    vsearch --cluster_size derep.fasta --id 0.98 --strand plus --sizein --sizeout --fasta_width 0 --centroids centroids.fasta
    """
}

process SwarmCluster{
    label 'swarm'
    publishDir "${params.out_dir}_results/merged_intermediates", mode: 'copy'

    input:
    path all_derep_fasta
    path centroids

    output:
    path "centroids.fasta"

    script:
    """
    swarm ${all_derep_fasta} --differences 1 --fastidious --seeds ${centroids} --usearch-abundance --output /dev/null
    """
}

process SingletonRemoval{
    label 'vsearch'
    publishDir "${params.out_dir}_results/merged_intermediates", mode: 'copy'

    input:
    path swarm_centroids

    output:
    path "sorted.fasta"

    script:
    """
    vsearch --sortbysize ${swarm_centroids} --sizein --sizeout --fasta_width 0 --minsize 2 --output sorted.fasta
    """
}

process DenovoChimeraRemoval{
    label 'vsearch'
    publishDir "${params.out_dir}_results/merged_intermediates", mode: 'copy'

    input:
    path nonsingleton_clusters

    output:
    path "denovo.nonchimeras.fasta"

    script:
    """
    vsearch --uchime_denovo ${nonsingleton_clusters} --sizein --sizeout --fasta_width 0 --qmask none --nonchimeras denovo.nonchimeras.fasta
    """
}

process ReferenceChimeraRemoval{
    label 'vsearch'
    publishDir "${params.out_dir}_results/merged_intermediates", mode: 'copy'

    input:
    path denovo_nonchimeras

    output:
    path "nonchimeras.fasta"

    script:
    """
    vsearch --uchime_ref ${denovo_nonchimeras} --db ${params.db_path} --sizein --sizeout --fasta_width 0 --qmask none --dbmask none --nonchimeras nonchimeras.fasta
    """
}

/*
    ================================================================================
                                        OTUs
    ================================================================================
*/

process RelabelOTUs{
    label 'vsearch'
    publishDir "${params.out_dir}_results/OTU_tables", mode: 'copy'

    input:
    path ref_nonchimeras

    output:
    path "otus.fasta"

    script:
    """
    vsearch --fastx_filter ${ref_nonchimeras} --sizein --sizeout --fasta_width 0 --relabel OTU_ --fastaout otus.fasta
    """
}

process MapOTUs{
    label 'vsearch'
    publishDir "${params.out_dir}_results/OTU_tables", mode: 'copy'

    input:
    path all_merged_fasta
    path relabeled_otus

    output:
    path "otutab.txt"

    script:
    """
    vsearch --usearch_global ${all_merged_fasta} --db ${relabeled_otus} --id 0.98 \
        --strand plus --sizein --sizeout --fasta_width 0 --qmask none --dbmask none \
        --otutabout otutab.txt 
    """
}

process ClassifyOTUs{
    label 'vsearch'
    publishDir "${params.out_dir}_results/OTU_tables", mode: 'copy'

    input:
    path relabeled_otus

    output:
    path "reads.sintax"

    script:
    """
    vsearch -sintax ${relabeled_otus} -db ${params.rdp_path} -tabbedout reads.sintax \
        -strand both --sintax_cutoff 0.8
    """
}

/*
    ================================================================================
                                OTU Table Modification
    ================================================================================
*/

process RelativeAbundance{
    label 'python'
    publishDir "${params.out_dir}_results/OTU_tables", mode: 'copy'

    input:
    path otu_table

    output:
    path "otutab_relative.txt"

    script:
    """
    python3 ${params.python_paths}count_relative.py $otu_table > otutab_relative.txt
    """
}

process MapOTUNames{
    label 'python'
    publishDir "${params.out_dir}_results/OTU_tables", mode: 'copy'

    input:
    path relative_abund
    path sintax_classification

    output:
    path "otutab_relative_withtaxa.txt"

    script:
    """
    python3 ${params.python_paths}otu_mapping.py $relative_abund $sintax_classification
    """
}

process MergeTaxa{
    label 'python'
    publishDir "${params.out_dir}_results/OTU_tables", mode: 'copy'

    input:
    path mapped_relative_otus

    output:
    path "otutab_relative_withtaxa_merged.tsv"

    script:
    """
    python3 ${params.python_paths}merge_abundance.py $mapped_relative_otus
    """
}

workflow {
    read_files = params.single_end ?
    Channel.fromPath("${params.in_dir}/*.fastq.gz")
        .map { file -> [file.baseName, file] } :
    Channel.fromFilePairs("${params.in_dir}/*_L001_R{1,2}_001.fastq.gz", size: 2)

    // Preprocessing
    if (!params.skip_cutadapt) {
        CutAdapt(read_files)
        FastQC(CutAdapt.out.trimmed_reads)
        if (!params.single_end) {
            vsearch_merge_out = VsearchMerge(CutAdapt.out.trimmed_reads)
        } else {
            vsearch_merge_out = CutAdapt.out.trimmed_reads
        }
    } else {
        FastQC(read_files)
        if (!params.single_end) {
            vsearch_merge_out = VsearchMerge(read_files)
        } else {
            vsearch_merge_out = read_files
        }
    }

    VsearchStats(vsearch_merge_out.merged_fastq)
    vsearch_filter_out = VsearchFilter(vsearch_merge_out.merged_fastq)
    derep_out = VsearchDereplicate(vsearch_filter_out)
    derep_fasta_files = derep_out.collect()
    all_merged_fasta = MergeAll(derep_fasta_files)
    all_derep_fasta = VsearchDerepAll(all_merged_fasta)

    //Cluster
    centroids = VsearchCluster(all_derep_fasta)
    swarm_centroids = SwarmCluster(all_derep_fasta, centroids)
    nonsingleton_clusters = SingletonRemoval(swarm_centroids)
    denovo_nonchimeras = DenovoChimeraRemoval(nonsingleton_clusters)
    ref_nonchimeras = ReferenceChimeraRemoval(denovo_nonchimeras)

    //OTUs
    relabeled_otus = RelabelOTUs(ref_nonchimeras)
    otu_table = MapOTUs(all_merged_fasta, relabeled_otus)
    sintax_classification = ClassifyOTUs(relabeled_otus)

    //OTU Table Modification
    relative_abund = RelativeAbundance(otu_table)
    mapped_relative_otus = MapOTUNames(relative_abund, sintax_classification)
    merged_otus = MergeTaxa(mapped_relative_otus)

}
