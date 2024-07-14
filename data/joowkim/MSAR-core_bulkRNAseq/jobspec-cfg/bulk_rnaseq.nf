nextflow.enable.dsl=2

process fastqc {
    tag "${meta.sample_name}"
    label "process_medium"

    publishDir "${projectDir}/analysis/fastqc/"

    module 'FastQC/0.11.9'

    input:
    tuple val(meta), path(reads)

    output:
    path ("*.zip"), emit: zips
    path ("*.html"), emit: htmls

    script:
    """
    fastqc --threads ${task.cpus} ${reads}
    """
}

process fastp {
    tag "${meta.sample_name}"
    label "process_medium"

    publishDir "${launchDir}/analysis/fastp/", mode: "copy"

    module 'fastp/0.21.0'

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta.sample_name), path("${meta.sample_name}_trimmed_R{1,2}.fastq.gz"), val(meta.single_end), emit: trim_reads
    path("${meta.sample_name}.fastp.json"), emit: fastp_json

    script:
    def adapter = "/mnt/beegfs/kimj32/reference/adapters.fa"
    if(!meta.single_end) {
    """
    fastp \
        -i ${reads[0]} \
        -I ${reads[1]} \
        --thread ${task.cpus} \
        --qualified_quality_phred 20 \
        -o ${meta.sample_name}_trimmed_R1.fastq.gz \
        -O ${meta.sample_name}_trimmed_R2.fastq.gz \
        --adapter_fasta $adapter \
        --json ${meta.sample_name}.fastp.json
    """
    } else {
    """
    fastp \
        -i ${reads} \
        --thread ${task.cpus} \
        --qualified_quality_phred 20 \
        -o ${meta.sample_name}_trimmed_R1.fastq.gz \
        --adapter_fasta $adapter \
        --json ${meta.sample_name}.fastp.json
    """
    }
}

process ribo_detector {
    tag "${sample_name}"
    label "memory_medium"

    publishDir "${launchDir}/analysis/ribo_detector"

    module 'RiboDetector/0.2.7'

    input:
    tuple val(sample_name), path(reads), val(is_SE)

    output:
    tuple val(sample_name), path("${sample_name}_rRNA_filt_R{1,2}.fastq.gz"), val(is_SE), emit: rRNA_filt_reads

    script:
    if (!is_SE) {
    """
    ribodetector_cpu -t ${task.cpus} \
        -i ${reads[0]} ${reads[1]} \
        -l 150 \
        -e rrna \
        -o ${sample_name}_rRNA_filt_R1.fastq.gz ${sample_name}_rRNA_filt_R2.fastq.gz
    """
    } else {
    """
    ribodetector_cpu -t ${task.cpus} \
        -i ${reads[0]} ${reads[1]} \
        -l 150 \
        -e rRNA \
        -o ${sample_name}_rRNA_filt_R1.fastq.gz \
        --log ${sample_name}.log
    """
    }
}


process star {
    tag "${sample_name}"
    label "memory_high"

    publishDir "${projectDir}/analysis/star/", mode : "copy"

    module "STAR/2.7.10a"
    //module 'samtools/1.16.1'

    input:
    tuple val(sample_name), path(reads), val(is_SE)

    output:
    tuple val(sample_name), path("${sample_name}.Aligned.sortedByCoord.out.bam"), val(is_SE), emit: bam
    //tuple val(sample_name), path("${sample_name}.Aligned.sortedByCoord.out.bam.bai"), emit: bai
    path("${sample_name}.Log.final.out"), emit: log_final
    path("${sample_name}.Log.out"), emit: log_out
    path("${sample_name}.ReadsPerGene.out.tab"), emit: read_per_gene_out
    path("${sample_name}.SJ.out.tab"), emit: sj_out
    path("${sample_name}._STAR*"), emit: out_dir // STARgenome and STARpass1
    path("${sample_name}.log"), emit: star_log_out // STARgenome and STARpass1
    tuple val(sample_name), path("${sample_name}_Unmapped_R{1,2}.fastq.gz"), val(is_SE), emit: unmapped_reads

    script:
    index = params.star_index.(params.genome)
    """
    STAR \
        --runThreadN ${task.cpus} \
        --genomeDir ${index} \
        --readFilesIn ${reads} \
        --twopassMode Basic \
        --readFilesCommand zcat \
        --outSAMtype BAM SortedByCoordinate \
        --outFileNamePrefix ${sample_name}. \
        --quantMode GeneCounts \
        --outReadsUnmapped Fastx \
        --outStd Log 2> ${sample_name}.log

    gzip -c "${sample_name}.Unmapped.out.mate1" > "${sample_name}_Unmapped_R1.fastq.gz"
    gzip -c "${sample_name}.Unmapped.out.mate2" > "${sample_name}_Unmapped_R2.fastq.gz"
    """
    // samtools index ${sample_name}.Aligned.sortedByCoord.out.bam
}


process preseq {
    tag "${sample_name}"
    label "memory_medium"

    publishDir "${projectDir}/analysis/preseq/"

    //module 'preseq/3.2.0'

    input:
    tuple val(sample_name), path(bam), val(is_SE)

    output:
    path("${sample_name}_{lc,c}_*.txt"), emit: preseq_output

    script:

    if (!is_SE) {
    """
    preseq lc_extrap -B -o ${sample_name}_lc_extrap.txt ${bam} -P -l 1000000000
    preseq c_curve -B -o ${sample_name}_c_curve.txt ${bam} -P -l 1000000000
    """
    } else {
    """
    preseq lc_extrap -B -o ${sample_name}_lc_extrap.txt ${bam} -l 1000000000
    preseq c_curve -B -o ${sample_name}_c_curve.txt ${bam} -l 1000000000
    """
    }
}


process samtools_index {
    tag "${sample_name}"
    label "process_low"

    publishDir "${projectDir}/analysis/samtools_index/"

    input:
    tuple val(sample_name), path(bam), val(is_SE)

    output:
    tuple val(sample_name), path("${sample_name}.Aligned.sortedByCoord.out.bam"), val(is_SE), emit : bam
    path("${sample_name}.Aligned.sortedByCoord.out.bam.bai")

    script:
    """
    samtools index ${bam}
    """
}

process qualimap {
    tag "${sample_name}"
    label "memory_medium"

    publishDir "${projectDir}/analysis/qualimap/"

    module 'qualimap/2.2.1'

    input:
    tuple val(sample_name), path(bam), val(is_SE)

    output:
    path("*"), emit: qualimap_out

    script:
    gtf = params.gtf.(params.genome)

    if (!is_SE) {
        // if sample is paired end data
    """
    qualimap rnaseq -bam ${bam} \\
        -gtf ${gtf} \\
        --paired \\
        -outdir quailmap_${sample_name} \\
        --java-mem-size=6G \\
        --sequencing-protocol strand-specific-reverse
    """
    } else {
        // if sample is single end data
    """
    qualimap rnaseq -bam ${bam} \\
        -gtf ${gtf} \\
        -outdir quailmap_${sample_name} \\
        --java-mem-size=6G \\
        --sequencing-protocol strand-specific-reverse
    """
    } // end if else
} // end process

process salmon {
    tag "${sample_name}"
    label "memory_medium"

    module "salmon/1.9"
    publishDir "${projectDir}/analysis/salmon/"

    input:
    tuple val(sample_name), path(reads), val(is_SE)

    output:
    path("${sample_name}"), emit: salmon_out

    script:
    // this is adapted from https://github.com/ATpoint/rnaseq_preprocess/blob/99e3d9b556325d2619e6b28b9531bf97a1542d3d/modules/quant.nf#L29
    def is_paired = is_SE ? "single" : "paired"
    def add_gcBias = is_SE ? "" : "--gcBias "
    def use_reads = is_SE ? "-r ${reads}" : "-1 ${reads[0]} -2 ${reads[1]}"
    index = params.salmon_index.(params.genome)
    """
    salmon quant \
        -p ${task.cpus} \
        -l A \
        -i ${index} \
        $use_reads \
        $add_gcBias \
        --validateMappings \
        -o ${sample_name}
    """
}

process seqtk {
    tag "${sample_name}"
    label "process_low"

    publishDir "${projectDir}/analysis/seqtk/"

    module 'seqtk/1.3'

    input:
    tuple val(sample_name), path(reads), val(is_SE)

    output:
    tuple val(sample_name), path("${sample_name}.subsample.100000.R{1,2}.fq.gz"), val(is_SE), emit: subsample_reads

    script:
    if(!is_SE) {
    """
    seqtk sample -s 100 ${reads[0]} 100000 | gzip -c > ${sample_name}.subsample.100000.R1.fq.gz
    seqtk sample -s 100 ${reads[1]} 100000 | gzip -c > ${sample_name}.subsample.100000.R2.fq.gz
    """
    } else {
    """
    seqtk sample -s 100 ${reads} 100000 | gzip -c > ${sample_name}.subsample.100000.R1.fq.gz
    """
    }
}

process sortMeRNA {
    tag "${sample_name}"
    label "process_medium"

    publishDir "${projectDir}/analysis/sortMeRNA/"

    module 'sortmerna/4.3.6'

    input:
    tuple val(sample_name), path(reads), val(is_SE)

    output:
    path ("*"), emit: sortMeRNA_out

    script:
    def threads = task.cpus
    def idx = "/mnt/beegfs/kimj32/tools/sortmerna/idx"
    def rfam5_8s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/rfam-5.8s-database-id98.fasta"
    def rfam5s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/rfam-5s-database-id98.fasta"
    def silva_arc_16s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/silva-arc-16s-id95.fasta"
    def silva_arc_23s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/silva-arc-23s-id98.fasta"
    def silva_euk_18s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/silva-euk-18s-id95.fasta"
    def silva_euk_28s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/silva-euk-28s-id98.fasta"
    def silva_bac_16s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/silva-bac-16s-id90.fasta"
    def silva_bac_23s = "/mnt/beegfs/kimj32/tools/sortmerna/data/rRNA_databases/silva-bac-23s-id98.fasta"

    if(!is_SE) {
     """
     sortmerna \\
        --threads ${threads} \\
        -reads ${reads[0]} \\
        --workdir sortMeRNA_${sample_name}  \\
        --ref ${rfam5s}  \\
        --ref ${rfam5_8s}  \\
        --ref ${silva_arc_16s}  \\
        --ref ${silva_arc_23s}  \\
        --ref ${silva_bac_16s}  \\
        --ref ${silva_bac_23s}  \\
        --ref ${silva_euk_18s}  \\
        --ref ${silva_euk_28s}
     """
    } else {
     """
     sortmerna \\
        --threads ${threads} \
        -reads ${reads} \
        --workdir sortMeRNA_${sample_name}  \
        --ref ${rfam5s}  \
        --ref ${rfam5_8s}  \
        --ref ${silva_arc_16s}  \
        --ref ${silva_arc_23s}  \
        --ref ${silva_bac_16s}  \
        --ref ${silva_bac_23s}  \
        --ref ${silva_euk_18s}  \
        --ref ${silva_euk_28s}
     """
    }
     // --idx-dir ${idx}  \\
}

process fastq_screen {
    tag "${sample_name}"
    label "process_medium"

    cpus 8
    memory '16 GB'

    publishDir "${projectDir}/analysis/fastq_screen"

    module 'FastQScreen/0.14.1'
    module 'bowtie2/2.3.4.1'

    input:
    tuple val(sample_name), path(reads), val(is_SE)

    output:
    path("*.html")
    path("*.txt"), emit: "fastq_screen_out"

    // threads option is already defined in fastq_screeN_conf
    script:
    def conf = params.fastq_screen_conf
    if(!is_SE) {
    """
    fastq_screen --aligner bowtie2 \
        --conf ${conf} \
        ${reads[0]} \
        --outdir ./
    """
    } else {
    """
    fastq_screen --aligner bowtie2 \
        --conf ${conf} \
        ${reads} \
        --outdir ./
    """
    }
}

process multiqc {
    label "process_dual"
    time "1h"

    publishDir "${projectDir}/analysis/multiqc/", mode : "copy"

    //module 'python/3.6.2'

    input:
    path(files)

    output:
    path("*.html"), emit: multiqc_output

    script:
    config_yaml = "/home/kimj32/config_defaults.yaml"
    """
    multiqc ${files} --filename "multiqc_report.html" --ignore '*STARpass1' --config ${config_yaml}
    """
}

process tpm_calculator {
    tag "tpm_calculator on ${sample_name}"
    label "process_medium"

    publishDir (
        path: "${projectDir}/analysis/tpm_calculator/",
        saveAs: {
            fn -> fn.replaceAll(".Aligned.sortedByCoord.out_genes", "")
        }
    )
    module "TPMCalculator/0.4.0"

    input:
    tuple val(sample_name), path(bam_file), val(is_SE)

    output:
    path("*out"), emit : "tpm_calculator_out"

    script:
    // -p option is for paired end data
    gtf = params.gtf.(params.genome)

    if ( ! is_SE ) {
    """
    TPMCalculator -g ${gtf} \
        -b ${bam_file} \
        -p
    """
    } else {
    """
    TPMCalculator -g ${gtf} \
        -b ${bam_file} \
    """
    }
}

// process kraken2{
//     tag "${sample_name}"
//     cpus 12
//     memory "84 GB"
//
//     publishDir "${projectDir}/analysis/kraken2/"
//
//     module "kraken/2.1.2"
//
//     input:
//     tuple val(sample_name), path(reads), val(is_SE)
//
//     output:
//
// }
/* process trim_galore {
    //debug true
    tag "${meta.sample_name}"
    cpus 4
    memory '8 GB'
    time "2h"

    publishDir "${projectDir}/analysis/trim_galore/"

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta.sample_name), path("*.gz"), val(meta.single_end), emit: trim_reads
    path("*.html")
    path("*.zip")
    path("*.txt")

    script:
    def threads = task.cpus - 1
    if(!meta.single_end) {
    """
    trim_galore \
    --paired \
    ${reads} \
    --cores ${threads} \
    -q 20 \
    --fastqc \
    --output_dir ./
    """
    } else {
    """
    trim_galore \
    ${reads[0]} \
    --cores ${task.cpus} \
    -q 20 \
    --fastqc \
    --output_dir ./
    """
    }
} */

log.info """
bulkRNAseq Nextflow
=============================================
samplesheet                           : ${params.samplesheet}
reference                       : ${params.ref_fa.(params.genome)}
"""

reference = Channel.fromPath(params.ref_fa.(params.genome), checkIfExists: true)

// See https://bioinformatics.stackexchange.com/questions/20227/how-does-one-account-for-both-single-end-and-paired-end-reads-as-input-in-a-next
ch_samplesheet = Channel.fromPath(params.samplesheet, checkIfExists: true)

// adapted from https://bioinformatics.stackexchange.com/questions/20227/how-does-one-account-for-both-single-end-and-paired-end-reads-as-input-in-a-next
ch_reads = ch_samplesheet.splitCsv(header:true).map {

    // This is the read1 and read2 entry
    r1 = it['fq1']
    r2 = it['fq2']

    // Detect wiether single-end or paired-end
    is_singleEnd = r2.toString() =='' ? true : false

    // The "meta" map, which is a Nextflow/Groovy map with id (the sample name) and a single_end logical entry
    meta = [sample_name: it['sample'], single_end: is_singleEnd]

    // We return a nested map, the first entry is the meta map, the second one is the read(s)
    r2.toString()=='' ? [meta, [r1]] : [meta, [r1, r2]]

}

workflow {

    fastqc(ch_reads)
    // trim_galore(ch_reads)
    fastp(ch_reads)
    seqtk(fastp.out.trim_reads)
    fastq_screen(seqtk.out.subsample_reads)
    sortMeRNA(seqtk.out.subsample_reads)

    if (params.run_ribodetector) {
    ribo_detector(fastp.out.trim_reads)
    star(ribo_detector.out.rRNA_filt_reads)
    samtools_index(star.out.bam)

    preseq(samtools_index.out.bam)
    qualimap(samtools_index.out.bam)

    } else {

    star(fastp.out.trim_reads)
    samtools_index(star.out.bam)

    preseq(samtools_index.out.bam)
    qualimap(samtools_index.out.bam)
    }

    if (params.run_salmon) {
        salmon(fastp.out.trim_reads)
        multiqc(qualimap.out.qualimap_out.mix(sortMeRNA.out.sortMeRNA_out, salmon.out.salmon_out, star.out.log_final, star.out.read_per_gene_out, fastqc.out.zips, fastq_screen.out.fastq_screen_out, fastp.out.fastp_json, preseq.out.preseq_output).collect())
    } else {
        multiqc( qualimap.out.qualimap_out.mix(sortMeRNA.out.sortMeRNA_out, star.out.log_final, star.out.read_per_gene_out, fastqc.out.zips, fastq_screen.out.fastq_screen_out, fastp.out.fastp_json, preseq.out.preseq_output).collect() )
    }

    if (params.run_tpm_calculator) {
        tpm_calculator(star.out.bam)
    }
}

workflow.onComplete {
	println ( workflow.success ? "\nDone!" : "Oops .. something went wrong" )
}
