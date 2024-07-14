#!/usr/bin/env nextflow

nextflow.enable.dsl=2

/*  ======================================================================================================
 *  HELP MENU
 *  ======================================================================================================
 */
line="=".multiply(100)
ver="qcflow-rnaseq v1.0.0"

// General params
params.help                  = null
params.output_dir            = "results"
params.genome                = null
params.genes                 = null
params.fastq_dir             = "data"

// system specific parameters
params.max_cpus              = 1
params.max_memory            = 1

config_profile_description   = null 
config_profile_contact       = null
config_profile_url           = null

params.workflow              = "all"
params.aligner	             = "star"

// STAR index
params.sjOverhang            = 149
params.snps                  = null

// STAR align 
params.library_name          = 0
params.index_dir             = null

// HISAT index
params.hisat_prefix          = "hisat_index"

// Fastp params
params.adapters              = "$PWD/data/truseq_adapters.fasta"
params.qual_phred            = 20
params.min_read_length       = 50

// Infer experiment params
params.cdna                  = null

params.strandedness          = "RF"

//--------------------------------------------------------------------------------------------------------
// Validation

include { validateParameters; paramsHelp; paramsSummaryLog; fromSamplesheet } from 'plugin/nf-validation'

if (params.help) {
   log.info paramsHelp("nextflow run ./main.nf ...")
   exit 0
}

// Validate input parameters
validateParameters()

// Print summary of supplied parameters
log.info paramsSummaryLog(workflow)

//-----------------------------------------------------------------------------------------------------------------------
// Functions

def extractCharacters(String inputString, library_name) {

    def intervals = library_name.split(',').collect { it.toInteger() }

    def adjustedStart
    def adjustedEnd

    if (intervals.size() == 1) {
        // Single value provided, create 'end'
        adjustedStart = 0
        adjustedEnd = inputString.length()
    } else if (intervals.size() == 2) {
        // Two values provided, use them as 'start' and 'end'
        adjustedStart = intervals[0] - 1
        adjustedEnd = intervals[1]
    } else {
        // Invalid input, handle appropriately (e.g., raise an error)
        throw new IllegalArgumentException("Invalid input provided: $library_name")
    }
    return inputString.substring(adjustedStart, adjustedEnd)
}

def is_null(var) {
  if (var == null) {
    return true
  } else {
    return false
  }
}

def selectTool(inputParameter) {
   selectedTool = ""
   if (inputParameter  ==~ /.*star.*/ ) {
        selectedTool = "star"
   } else if (inputParameter ==~ /.*hisat.*/) {
        selectedTool = "hisat"
   } else {
        selectedTool = "reads"
   }
   return selectedTool
}

workflow_input = params.workflow

switch (workflow_input) {
    case ["genome-index"]:
        include { run_star_index; run_star_index_snps; run_hisat_index; run_hisat_index_high_mem } from './modules/module_prep_index.nf'
        aligner = params.aligner
	genome = file(params.genome)
        genes = file(params.genes)
	output_dir = params.output_dir
	snp = params.snps
        break;
     case ["reads-qc"]:
	include { run_fastqc; run_multiqc_reads } from './modules/module_read_qc.nf'
	fastq_dir = params.fastq_dir
        samples = Channel.fromFilePairs("${fastq_dir}", type: 'file')
                    .ifEmpty { exit 1, fastq_dir }
        break;
     case ["reads-qc-cont"]:
	include { run_fastqc; run_multiqc_reads } from './modules/module_read_qc.nf'
	include { run_cont; combine_cont_bbt } from './modules/module_read_cont.nf'
	fastq_dir = params.fastq_dir
	bbt_filters = params.bbt_filters
	samples = Channel.fromFilePairs("${fastq_dir}", type: 'file', checkIfExists: true)
	bbt_filters = Channel.fromPath("${bbt_filters}", type: 'file')
		.filter { file -> file.name.endsWith('.bf') }
        break;
    case ["trim"]:
        include { run_fastp; run_fastqc; run_multiqc_trimming } from './modules/module_read_trimming.nf'
        adapters = file(params.adapters)
        fastq_dir = params.fastq_dir
        samples = Channel.fromFilePairs("${fastq_dir}", type: 'file')
                    .ifEmpty { exit 1, fastq_dir }
        break;
     case ["align"]:
	include { run_star_align_plants; run_star_align; run_hisat_align; run_multiqc_align } from './modules/module_read_align.nf'
	include { convert_bed; run_bam_stats; run_junction_annotation; combine_bam_stats } from './modules/module_align_qc.nf'
	include { combine_counts_star; combine_counts_featurecounts; run_feature_counts } from './modules/module_align_counts.nf'
	strandedness = params.strandedness
	fastq_dir = params.fastq_dir
        library_name = params.library_name
	genome = params.genome
	genes = file(params.genes)
	index = params.index_dir
	aligner = params.aligner
	samples_align = Channel.fromFilePairs(fastq_dir, flat: true)
		.map { prefix, file1, file2 -> tuple(extractCharacters(prefix,library_name), file1, file2) }
                .groupTuple(sort: true)
                .ifEmpty { exit 1, fastq_dir }
	break;
      case ["infer-strandedness"]:
	include { run_check_strandedness } from './modules/module_align_qc.nf'
	cdna = params.cdna
	fastq_dir = params.fastq_dir
	genes = file(params.genes)
	samples = Channel.fromFilePairs("${fastq_dir}", type:'file', size: 2)
		.ifEmpty { exit 1, fastq_dir }
	break;
}


workflow GENOME_INDEX {
    take:
    genome
    genes

    main:

    if (aligner == "hisat-highmem") {
	run_hisat_index_high_mem(genome, genes)
     } else if (aligner == "hisat") {
	run_hisat_index(genome, genes) 
     } else if (aligner == "star" || aligner == "star-plants")  {
	run_star_index(genome, genes)
     } else if (aligner == "star-snps") {
	run_star_index_snps(genome, genes)
     }
}

workflow READ_QC {
    take:
    samples
       
    main:
    run_fastqc(samples)
    run_fastqc.out.qc_html
        .map { it -> it[1] }
        .collect()
        .set { fastqc_out }
    run_multiqc_reads(fastqc_out)
}

workflow READ_QC_CONT {
    take:
    samples
    bbt_filters

    main:
    bbt_filters
	.collect()
	.set { bbt }
    output_cont = run_cont(samples, bbt)

    output_cont.cont_out
        .map { it -> it[1]}
        .flatten()
        .collect()
        .set { cont_counts }
    combine_cont_bbt(cont_counts)

    READ_QC(samples)
}

workflow TRIM_READS {
    take:
    samples
    adapters

    main:
    fastp_out = run_fastp(samples, adapters)
    fastp_out.json
        .map { it -> it[1] }
        .collect()
        .set { fastp_json }

    fastqc_trimmed_out = run_fastqc(fastp_out.trimmed_reads)
    run_multiqc_trimming(fastp_json.mix(fastqc_trimmed_out).collect())
}

workflow ALIGN_READS {
     take:
     samples_align
     aligner
     genes

     main:

     if (aligner == "star" || aligner == "star-snps") {
	output_align = run_star_align(samples_align, index, genes)
     } else if (aligner == "star-plants") {
	output_align = run_star_align_plants(samples_align, index, genes)
     } else if (aligner ==~ /.*hisat.*/)  {
	output_align = run_hisat_align(samples_align, index, genes)
     } 

     if (aligner ==~ /.*star.*/) {

	output_align.counts
        .map { it -> it[1]}
        .flatten()
        .collect()
        .set { counts }

	combine_counts_star(counts)

     } else if (aligner ==~ /.*hisat.*/) {

	output_counts = run_feature_counts(output_align.alignements, genome, genes)

	output_counts.counts
        .map { it -> it[1]}
        .flatten()
        .collect()
        .set { counts }

	combine_counts_featurecounts(counts)
     }

     // Create reports
     output_align.reports
        .map { it -> it[1]}
        .flatten()
        .collect()
        .set { reports }
     convert_bed(genes)

     // QC stages
     output_bam_stats = run_bam_stats(output_align.alignements, convert_bed.out)
     run_junction_annotation(output_align.alignements, convert_bed.out)
     run_multiqc_align(reports, selectTool(params.aligner))

     // Combine results for bam stats in a tabular format
     output_bam_stats.bam_stats
        .map { it -> it[1]}
        .flatten()
        .collect()
        .set { bam_stats }

     combine_bam_stats(bam_stats)
}

workflow INFER_STRANDEDNESS {
    take:
    cdna
    genes
    samples

    main:
    run_check_strandedness(genes, cdna, samples)
}

workflow {
	switchVariable = 0
	
	if (workflow_input == "genome-index") {
		switchVariable = 1;
	} else if (workflow_input == "reads-qc") {
		switchVariable = 2;
	} else if (workflow_input == "reads-qc-cont") {
		switchVariable = 3;
	} else if (workflow_input == "trim") {
		switchVariable = 4;
	} else if (workflow_input == "align") {
		switchVariable = 5;
	} else if (workflow_input == "infer-strandedness") {
		switchVariable = 6;
	}
 
	switch (switchVariable) {
	case 1:
		GENOME_INDEX(genome, genes);
		break;
	case 2:
		READ_QC(samples);
		break;
	case 3:
		READ_QC_CONT(samples, bbt_filters);
                break;
	case 4:
		TRIM_READS(samples, adapters)
		break;
	case 5:
		ALIGN_READS(samples_align, aligner, genes)
		break;
	case 6:
		INFER_STRANDEDNESS(cdna, genes, samples)
		break;
	default:
		println("Please provide the correct input options")
		break;
	}
}
