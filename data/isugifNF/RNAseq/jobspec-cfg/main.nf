#! /usr/bin/env nextflow
/* Name: RNASeq pipeline
 * Auth: Jennifer Chang
 * Date: 2021/03/14
 * Desc: Given a reference genome and paired reads, create RNAseq counts files
 */

nextflow.enable.dsl=2

def helpMsg() {
  log.info """
   Usage:
   The typical command for running the RNAseq pipeline is as follows:
   nextflow run main.nf --reads "*_{R1,R2}.fastq.gz" --genome GENOME.fasta --genome_gff GENOME.gff --genome_cdna CDNA.fasta  -profile singularity

   Mandatory arguments:
    --reads                 Paired-end reads in fastq.gz format, will need to specify glob (e.g. "*_{R1,R2}.fastq.gz")
    --genome                Reference genome fasta file
    --genome_gff            Reference gff file
    --genome_cdna           Reference cdna file

   Optional configuration arguments:
    --methods               Select one or more RNAseq counting methods [default: 'gsnap,hisat2,kallisto,salmon']
    -profile                Configuration profile to use. Can use multiple (comma separated)
                            Available: local, slurm [default:local]
    --fastqc_app            Link to fastqc executable [default: 'fastqc']
    --multiqc_app           Link to multiqc executable [default: 'multiqc']
    --parallel_app          Link to parallel executable [default: 'parallel']
    --kallisto_app          Link to kallisto executable [default: 'kallisto']
    --salmon_app            Link to salmon executable [default: 'salmon']
    --gsnap_app             Link to gsnap executable [default: 'gsnap']
    --featureCounts_app     Link to featureCounts executable [default: 'featureCounts']

   Optional other arguments:
    --threads               Threads per process [default:4 for local, 16 for slurm]
    --queueSize             Maximum jobs to submit to slurm [default:20]
    --account               HPC account name for slurm sbatch, atlas and ceres requires this
    --help
"""
}

if(params.help){
  helpMsg()
  exit 0
}

// === Define Processes
process fastqc {
    tag "batched"
    label 'fastqc'

    publishDir "${params.outdir}/01_Quality-Control", mode: 'copy',
        saveAs: { filename -> filename.indexOf(".zip") > 0 ? "zips/$filename" : "$filename" }

    input: path(read)
    output: path("*_fastqc.{zip,html}")
    script:
    """
    #! /usr/bin/env bash
    PROC=\$((`nproc`))
    ${parallel_app} -j \${PROC} "${fastqc_app} {1}" ::: ${read}
    """
}

process multiqc {
    label 'multiqc'

    publishDir "${params.outdir}/01_Quality-Control", mode: 'copy',
        saveAs: { filename -> filename.indexOf(".zip") > 0 ? "zips/$filename" : "$filename" }

    input: path fastqc_htmls

    output: path "multiqc_report.html"

    script:
    """
    #! /usr/bin/env bash
    $multiqc_app .
    """
}

process kallisto_index {
    tag "$genome_cdna"
    label 'kallisto'

    publishDir "${params.outdir}/03_Kallisto", mode: 'copy'

    input: path(genome_cdna)

    output: path("${genome_cdna.simpleName}.idx")

    script:
    """
    #! /usr/bin/env bash
    $kallisto_app index \
      -i ${genome_cdna.simpleName}.idx ${genome_cdna}
    """
}

process kallisto_quant {
    tag "${readname}"
    label 'kallisto'

    publishDir "${params.outdir}/03_Kallisto", mode: 'copy'

    input: tuple path(genome_index), val(readname), path(read_pairs)

    output: path("*")

    script:
    """
    #! /usr/bin/env bash
    PROC=\$((`nproc`))
    ${kallisto_app} quant \
     -i ${genome_index} \
     -o ${readname}_quant \
     -b 20 -t \$PROC \
     ${read_pairs}
    """
}

process salmon_index {
    tag "$genome_cdna"
    label 'salmon'

    publishDir "${params.outdir}/03_Salmon", mode: 'copy'

    input: path(genome_cdna)

    output: tuple val("${genome_cdna.simpleName}"), path("*")

    script:
    """
    #! /usr/bin/env bash
    $salmon_app index \
      -i ${genome_cdna.simpleName} \
      -t ${genome_cdna}
    """
}

process salmon_quant {
    tag "${readname}"
    label 'salmon'

    publishDir "${params.outdir}/03_Salmon", mode: 'copy'

    input:
    tuple val(genome_index), path(genome_index_files), val(readname), path(read_pairs)

    output:
    path("*")

    script:
    """
    #! /usr/bin/env bash
    PROC=\$((`nproc`))
    $salmon_app quant \
     -l A -p \$PROC \
     --validateMappings \
     -i ${genome_index} \
     -1 ${read_pairs.get(0)} \
     -2 ${read_pairs.get(1)} \
     -o ${readname}_quant
    """
}

process gsnap_index {
    tag "${genome_gz.simpleName}"
    label 'gsnap'

    publishDir "${params.outdir}/03_GSNAP", mode: 'copy'

    input:
    path(genome_gz)

    output:
    tuple val("${genome_gz.simpleName}"), path("*")

    script:
    """
    #! /usr/bin/env bash
    gmap_build \
     --gunzip \
     -d ${genome_gz.simpleName} \
     -D gmapdb \
     ${genome_gz}
    """
}

process gsnap_align {
    tag "${readname}"
    label 'gsnap'

    publishDir "${params.outdir}/03_GSNAP", mode: 'copy'

    input: tuple val(genome_name), path(gmap_dir),val(readname), path(read_pairs)

    output: path("*")

    script:
    """
    #! /usr/bin/env bash
    PROC=\$(((`nproc`-1)*3/4+1))
    PROC2=\$(((`nproc`-1)*1/4+1))
    ${gsnap_app} \
     --gunzip \
     -d ${genome_name} \
     -D gmapdb/ \
     -N 1 -t \$PROC -B 4 -m 5 \
     --input-buffer-size=1000000 \
     --output-buffer-size=1000000 \
     -A sam \
     ${read_pairs} |
     samtools view --threads \$PROC -bS - > ${readname}.bam
    """
}

process featureCounts_gene {
    tag "${read_bam.simpleName}"
    label 'gsnap'

    publishDir "${params.outdir}/03_GSNAP", mode: 'copy'
    input: tuple path(read_bam), path(genome_gff)

    output: path("*")

    script:
    """
    #! /usr/bin/env bash
    PROC=\$((`nproc`))
    $featureCounts_app \
      -T \$PROC \
      -t gene \
      -g ID \
      -p \
      -a ${genome_gff} \
      -o ${read_bam.simpleName}_genecounts.txt \
      ${read_bam}
    """
}

process featureCounts_mRNA {
    tag "${read_bam.simpleName}"
    label 'gsnap'

    publishDir "${params.outdir}/03_GSNAP", mode: 'copy'

    input:
    tuple path(read_bam), path(genome_gff)

    output:
    path("*")

    script:
    """
    #! /usr/bin/env bash
    PROC=\$((`nproc`))
    $featureCounts_app \
      -T \$PROC -p \
      -t mRNA \
      -g ID \
      -a ${genome_gff} \
      -o ${read_bam.simpleName}_mRNAcounts.txt \
      ${read_bam}
    """
}

process featureCounts_geneMult {
    tag "${read_bam.simpleName}"
    label 'gsnap'

    publishDir "${params.outdir}/03_GSNAP", mode: 'copy'

    input:
    tuple path(read_bam), path(genome_gff)

    output:
    path("*")

    script:
    """
    #! /usr/bin/env bash
    PROC=\$((`nproc`))
    $featureCounts_app \
      -T \$PROC -M -p \
      -t gene \
      -g ID \
      -a ${genome_gff} \
      -o ${read_bam.simpleName}_geneMultcounts.txt \
      ${read_bam}
    """
}

// === Main Workflow
workflow {
  /* Read, reference, and gff channels */
  readsflat_ch = channel.fromPath(params.reads, checkIfExists:true)
  reads_ch = channel.fromFilePairs(params.reads, checkIfExists:true)

  if(params.methods =~ /gsnap/) {
    genome_ch = channel.fromPath(params.genome, checkIfExists:true)
    gff_ch = channel.fromPath(params.genome_gff, checkIfExists:true)
  }

  if(params.methods =~ /kallisto/ || params.methods =~ /salmon/){
    cdna_ch = channel.fromPath(params.genome_cdna, checkIfExists:true)
  }

  /* 01_Quality-Control */
  readsflat_ch.collate(8) | fastqc | collect | multiqc

  /* 03_Kallisto */
  if(params.methods =~ /kallisto/){
    cdna_ch | kallisto_index | combine(reads_ch) | kallisto_quant
  }

  /* 03_Salmon */
  if(params.methods =~ /salmon/){
    cdna_ch | salmon_index | combine(reads_ch) | salmon_quant
  }

  /* 03_GSNAP */
  if(params.methods =~ /gsnap/){
    genome_ch | gsnap_index | combine(reads_ch) | gsnap_align | combine(gff_ch) |
      ( featureCounts_gene & featureCounts_mRNA & featureCounts_geneMult)
  }
}
