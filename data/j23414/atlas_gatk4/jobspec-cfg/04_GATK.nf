#! /usr/bin/env nextflow
/* Name: GATK4 Pipeline
 * Auth: Jennifer Chang
 * Date: 2021/04/01
 * Desc:
 * Test:
 */

nextflow.enable.dsl=2

params.genome="00_Raw-Data/test-data/ref/*.fasta"
params.reads="00_Raw-Data/test-data/fastq/*_{R1,R2}.fastq.gz"
params.outdir="04_GATK"
params.reads_file=false
params.window=100000

// === Define Processes

process FastqToSam {
  tag "$readname"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "${params.outdir}", mode: 'copy'

  input:  // [readgroup, [left.fq.gz, right.fq.gz], increment_readgroup]
  tuple val(readname), path(readpairs), val(i_readname)

  output: // increment_readgroup.bam since one readgroup can have multiple lanes
  path("${i_readname}.bam")

  """
  #! /usr/bin/env bash
  gatk FastqToSam \
  --FASTQ ${readpairs.get(0)} \
  --FASTQ2 ${readpairs.get(1)} \
  --OUTPUT ${i_readname}.bam \
  --READ_GROUP_NAME ${readname} \
  --SAMPLE_NAME ${readname}_name \
  --LIBRARY_NAME ${readname}_lib \
  --PLATFORM ILLUMINA \
  --SEQUENCING_CENTER ISU
  """
}
// USE_JDK_DEFLATER=true USE_JDK_INFLATER=true

process MarkIlluminaAdapters {
  tag "${bam.fileName}"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "${params.outdir}", mode: 'copy'

  input:  // reads.bam
  path(bam)

  output: // reads_marked.bam
  path("${bam.simpleName}_marked.bam")

  script:
  """
  #! /usr/bin/env bash
  gatk MarkIlluminaAdapters \
  --INPUT ${bam} \
  --OUTPUT ${bam.simpleName}_marked.bam \
  --METRICS ${bam.simpleName}_marked_metrics.txt
  """
}

process SamToFastq {
  tag "${bam.fileName}"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "${params.outdir}", mode: 'copy'

  input:  // reads.bam
  path(bam)

  output: // reads_interleaved.fq
  tuple val("${bam.simpleName}"), path("${bam.simpleName}_newR1.fq"), path("${bam.simpleName}_newR2.fq")

  script:
  """
  #! /usr/bin/env bash
  gatk SamToFastq \
  --INPUT ${bam} \
  --FASTQ ${bam.simpleName}_newR1.fq \
  --SECOND_END_FASTQ ${bam.simpleName}_newR2.fq \
  --CLIPPING_ATTRIBUTE XT \
  --CLIPPING_ACTION 2 \
  --INCLUDE_NON_PF_READS true
  """
}
// --INTERLEAVE true

process bwamem2_index {
    tag "${genome_fasta.simpleName}"
    label 'bwamem'
    executor 'slurm'
    clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

    publishDir "${params.outdir}", mode: 'copy'

    input:
    path(genome_fasta)

    output: // [ genome.fasta, [genome index files] ]
    tuple path("$genome_fasta"), path("${genome_fasta}*")

    script:
    """
    #! /usr/bin/env bash
    bwa-mem2 index ${genome_fasta}
    """
}

process bwamem2_mem {
    tag "${readname}"
    label 'bwamem'
    executor 'slurm'
    clusterOptions '-N 1 -n 16 -t 04:00:00 --account=isu_gif_vrsc'

    publishDir "${params.outdir}", mode: 'copy'

    input:
    tuple path(genome_fasta), path(genome_index_files), \
      val(readname), path(readpairs)

    output:
    path("${readname}_mapped.bam")

    script:
    """
    #! /usr/bin/env bash
    bwa-mem2 mem -t 16 ${genome_fasta} ${readpairs} |\
     samtools view --threads 16 -bS - > ${readname}_mapped.bam
    """
}

process CreateSequenceDictionary {
  tag "$genome_fasta.simpleName"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'
  publishDir "${params.outdir}", mode: 'copy'

  input:
  path(genome_fasta)

  output: // genome.dict
  path("${genome_fasta.simpleName}.dict")

  script:
  """
  #! /usr/bin/env bash
  gatk CreateSequenceDictionary \
  -R ${genome_fasta} \
  -O ${genome_fasta.simpleName}.dict
  """
}

process samtools_faidx {
  tag "${genome_fasta.simpleName}"
  label 'samtools'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'
  publishDir "${params.outdir}", mode: 'copy'

  input:  // genome.fasta
  path(genome_fasta)

  output:
  path("*.fai")

  """
  #! /usr/bin/env bash
  samtools faidx $genome_fasta
  """
}

process MergeBamAlignment {
  tag "$i_readname"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "${params.outdir}", mode: 'copy'

  input:
  tuple val(i_readname), path(read_unmapped), path(read_mapped), path(genome_fasta), path(genome_dict)

  output: // merged bam and bai files
  tuple path("${i_readname}_merged.bam"), path("${i_readname}_merged.bai")

  script:
  """
  #! /usr/bin/env bash
  gatk MergeBamAlignment \
  --REFERENCE_SEQUENCE $genome_fasta \
  --UNMAPPED_BAM ${read_unmapped} \
  --ALIGNED_BAM ${read_mapped} \
  --OUTPUT ${i_readname}_merged.bam \
  --CREATE_INDEX true \
  --ADD_MATE_CIGAR true \
  --CLIP_ADAPTERS false \
  --CLIP_OVERLAPPING_READS true \
  --INCLUDE_SECONDARY_ALIGNMENTS true \
  --MAX_INSERTIONS_OR_DELETIONS -1 \
  --PRIMARY_ALIGNMENT_STRATEGY MostDistant \
  --ATTRIBUTES_TO_RETAIN XS
  """
}

process makewindows {
  tag "${genome_fasta.simpleName}"
  label 'samtools'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'
  publishDir "${params.outdir}", mode: 'copy'

  input:
  path(genome_fasta)

  output:
  path("${genome_fasta.simpleName}_coords.bed")

  """
  #! /usr/bin/env bash
  samtools faidx $genome_fasta
  awk -F'\t' '{print \$1"\t"\$2}' ${genome_fasta}.fai > genome_length.txt
  bedtools makewindows -w $params.window -g genome_length.txt |\
    awk '{print \$1"\t"\$2+1"\t"\$3}' |\
    sed \$'s/\t/:/1' |\
    sed \$'s/\t/-/g' > ${genome_fasta.simpleName}_coords.bed
  """
}

process gatk_HaplotypeCaller {
  tag "$window"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "${params.outdir}", mode: 'copy'

  input:  // [window, reads files ..., genome files ...]
  tuple val(window), path(bam), path(bai), path(genome_fasta), path(genome_dict), path(genome_fai)

  output: // identified SNPs as a vcf file
  path("*.vcf")

  script:
  """
  #! /usr/bin/env bash
  BAMFILES=`echo $bam | sed 's/ / -I /g' | tr '[' ' ' | tr ']' ' '`
  gatk --java-options \"-Xmx80g -XX:+UseParallelGC\" HaplotypeCaller \
  -R $genome_fasta \
  -I \$BAMFILES \
  -L $window \
  --output ${window.replace(':','_')}.vcf
  """
}

process merge_vcf {
  tag "merging"
  label 'merge'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "${params.outdir}", mode: 'copy'

  input:  // multiple SNP vcf files
  path(vcfs)

  output: // merged into one vcf file
  path "first-round_merged.vcf"

  script:
  """
  #! /usr/bin/env bash
  cat ${vcfs.get(0)} | grep "^#" > first-round_merged.vcf
  cat ${vcfs} | grep -v "^#" >> first-round_merged.vcf
  """
}

process vcftools_snp_only {
  tag "${merged_vcf.fileName}"
  label 'vcftools'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "${params.outdir}", mode: 'copy'

  input:  // merged SNP vcf file
  path(merged_vcf)

  output: // vcf file only containing SNPs
  path("${merged_vcf.simpleName}_snps-only.*")

  script:
  """
  #! /usr/bin/env bash
  vcftools \
  --vcf $merged_vcf \
  --remove-indels \
  --recode \
  --recode-INFO-all \
  --out ${merged_vcf.simpleName}_snps-only
  """
}

process SortVcf {
  tag "$vcf.fileName"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  publishDir "$params.outdir", mode: 'copy'

  input:  // [SNP.vcf, genome.dict]
  tuple path(vcf), path(dict)

  output: // sorted SNP.vcf
  path ("*.vcf")

  script:
  """
  #! /usr/bin/env bash
  gatk SortVcf \
  --INPUT $vcf \
  --SEQUENCE_DICTIONARY $dict \
  --CREATE_INDEX true \
  --OUTPUT ${vcf.simpleName}_sorted.vcf
  """
}

process calc_DPvalue {
  tag "$sorted_vcf.fileName"
  label 'datamash'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'

  input:  // sorted SNP vcf
  path(sorted_vcf)

  output: // DP value (number) to filter vcf in downstream process
  stdout()

  script:
  """
  #! /usr/bin/env bash
  grep -v "^#" $sorted_vcf | cut -f 8 | grep -oe ";DP=.*" | cut -f 2 -d ';' | cut -f 2 -d "=" > dp.txt
  cat dp.txt | datamash mean 1 sstdev 1 > dp.stats
  cat dp.stats | awk '{print \$1+5*\$2}'
  """
}

process VariantFiltration {
  tag "$sorted_snp_vcf.fileName"
  label 'gatk'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'
  publishDir "$params.outdir", mode: 'copy'

  input:  // [sorted snp vcf, DP filter, genome files ... ]
  tuple path(sorted_snp_vcf), val(dp), path(genome_fasta), path(genome_dict), path(genome_fai)

  output: // filtered to identified SNP variants
  path("${sorted_snp_vcf.simpleName}_marked.vcf")

  script:
  """
  #! /usr/bin/env bash
  gatk VariantFiltration \
    --reference $genome_fasta \
    --sequence-dictionary $genome_dict \
    --variant $sorted_snp_vcf \
    --filter-expression \"QD < 2.0 || FS > 60.0 || MQ < 40.0 || MQRankSum < -12.5 || ReadPosRankSum < -8.0 || DP > $dp\" \
    --filter-name "FAIL" \
    --output ${sorted_snp_vcf.simpleName}_marked.vcf
  """
}

process keep_only_pass {
  tag "$snp_marked_vcf.fileName"
  label 'keeppassed'
  executor 'slurm'
  clusterOptions '-N 1 -n 16 -t 02:00:00 --account=isu_gif_vrsc'
  publishDir "$params.outdir", mode: 'copy'

  input:
  path(snp_marked_vcf)

  output:
  path("${snp_marked_vcf.simpleName}_snp-only.pass-only.vcf")

  script:
  """
  #! /usr/bin/env bash
  cat $snp_marked_vcf | grep "^#" > ${snp_marked_vcf.simpleName}_snp-only.pass-only.vcf
  cat $snp_marked_vcf | grep -v "^#" | awk '\$7=="PASS"' >> ${snp_marked_vcf.simpleName}_snp-only.pass-only.vcf
  """
}

// === Main Workflow
workflow {
  genome_ch = channel.fromPath(params.genome, checkIfExists:true)
  if (params.reads) {
    reads_ch = channel.fromFilePairs(params.reads, checkIfExists:true)
  } else {
    reads_ch = channel.fromPath(params.reads_file, checkIfExists:true) |
    splitCsv(sep:'\t') |
    map { n -> [ n.get(0), [n.get(1), n.get(2)]] }
  }

  // == Since one sample may be run on multiple lanes
  i = 1
  ireads_ch = reads_ch | map { n -> [n.get(0), n.get(1), "${i++}_"+n.get(0)] }

  // == Prepare mapped and unmapped read files
  cleanreads_ch = ireads_ch | FastqToSam | MarkIlluminaAdapters | SamToFastq |
    map { n -> [ n.get(0).replaceFirst("_marked",""), [ n.get(1), n.get(2)] ] }

  genome_ch | bwamem2_index | combine(cleanreads_ch) | bwamem2_mem

  mapped_ch = bwamem2_mem.out |
    map { n -> [n.simpleName.replaceFirst("_mapped",""), n] }

  // Might be FastqToSam out
  unmapped_ch = MarkIlluminaAdapters.out |
    map { n -> [n.simpleName.replaceFirst("_marked",""), n] }

  genome_ch | ( CreateSequenceDictionary & samtools_faidx )
  unmapped_ch | join(mapped_ch) | combine(genome_ch) | combine(CreateSequenceDictionary.out) | MergeBamAlignment

  allbai_ch = MergeBamAlignment.out | map { n -> n.get(1)} | collect | map { n -> [n]}
  allbambai_ch = MergeBamAlignment.out | map { n -> n.get(0)} | collect | map { n -> [n]} | combine(allbai_ch)

  // == Run Gatk Haplotype by interval window
  genome_ch | makewindows | splitText( by:1) |
    map { n -> n.replaceFirst("\n","") } |
    combine(allbambai_ch) | 
    combine(genome_ch) |
    combine(CreateSequenceDictionary.out) |
    combine(samtools_faidx.out) |
    gatk_HaplotypeCaller |
    collect |
    merge_vcf |
    vcftools_snp_only |
    combine(CreateSequenceDictionary.out) |
    SortVcf |
    calc_DPvalue

  // == Filter resulting SNPs
  SortVcf.out |
    combine(calc_DPvalue.out.map{n-> n.replaceAll("\n","")}) |
    combine(genome_ch) |
    combine(CreateSequenceDictionary.out) |
    combine(samtools_faidx.out) |
    VariantFiltration |
    keep_only_pass
}
