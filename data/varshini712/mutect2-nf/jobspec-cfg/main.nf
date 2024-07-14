params.ref_dir = "ref"
params.gatk_bundle_dir = "${params.ref_dir}/gatk-bundle"

params.targetbed = "/gpfs/data/molecpathlab/development/mutect2-nf/targets.annotated.580.bed"
params.ref_fa = "${params.ref_dir}/iGenomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa"
params.ref_fai = "${params.ref_dir}/iGenomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa.fai"
params.ref_dict = "${params.ref_dir}/iGenomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.dict"
params.germline_resource_gz = "${params.gatk_bundle_dir}/af-only-gnomad.raw.sites.hg19.vcf.gz"
params.germline_resource_gz_tbi = "${params.gatk_bundle_dir}/af-only-gnomad.raw.sites.hg19.vcf.gz.tbi"

params.samplesheet = "sample.pairs.tsv"
params.outputdir = "output"

// Set up input data channels for targets, ref and germline vcf files
Channel.fromPath( file(params.targetbed) ).into { targets_bed; targets_bed2 }

Channel.fromPath( file(params.ref_fa) ).into { ref_fasta; ref_fasta2 }
Channel.fromPath( file(params.ref_fai) ).into { ref_fai; ref_fai2 }
Channel.fromPath( file(params.ref_dict) ).into { ref_dict; ref_dict2 }

Channel.fromPath( file(params.germline_resource_gz) ).set { germline_resource_gz }
Channel.fromPath( file(params.germline_resource_gz_tbi) ).set { germline_resource_gz_tbi }

// Read in sample paris file with sample id and bams
Channel.fromPath( file(params.samplesheet) )
       .splitCsv(header: true, sep: '\t')
       .map{row ->
         def sampleID = row['Sample']
         def tumorID = row['Tumor']
         def normalID = row['Normal']
         def tumorBam = row['Tumor_Bam'].tokenize( ',' ).collect { file(it) }
         def tumorBai = row['Tumor_Bai'].tokenize( ',' ).collect { file(it) }
         def normalBam = row['Normal_Bam'].tokenize( ',' ).collect { file(it) }
         def normalBai = row['Normal_Bai'].tokenize( ',' ).collect { file(it) }
         return [ sampleID, tumorID, normalID, tumorBam, tumorBai, normalBam, normalBai ]
       }
       .tap { samples_bam_bai;  samples_bam_bai2}

samples_bam_bai.combine(ref_fasta)
               .combine(ref_fai)
               .combine(ref_dict)
               .combine(targets_bed)
               .combine(germline_resource_gz)
               .combine(germline_resource_gz_tbi)
               .set { sample_bam_pairs_ref_germlinevcf }

//sample_bam_pairs_ref_germlinevcf.subscribe { println "value $it" }
process mutect2 {

    publishDir "${params.outputdir}/mutect2", mode: 'copy', overwrite: true
    echo true

    input:
    set val(sampleID), val(tumorID), val(normalID), file(tumorBam), file(tumorBai), file(normalBam), file(normalBai), file(ref_fasta), file(ref_fai), file(ref_dict), file(targets_bed), file(germline_resource_gz), file(germline_resource_gz_tbi) from sample_bam_pairs_ref_germlinevcf

    output:
    file("${vcf_file}")

    script:
    prefix = "${tumorID}.${normalID}"
    vcf_file = "${prefix}.vcf"

    """
    gatk --java-options \"-Xms8G -Xmx8G\" Mutect2 \
    --seconds-between-progress-updates 600 \
    --native-pair-hmm-threads 4 \
    --reference "${ref_fasta}" \
    --germline-resource "${germline_resource_gz}" \
    --dont-use-soft-clipped-bases \
    --max-reads-per-alignment-start 100 \
    --intervals "${targets_bed}" \
    --interval-padding 10 \
    --input "${tumorBam}" \
    --input "${normalBam}" \
    --tumor-sample "${tumorID}" \
    --normal-sample "${normalID}" \
    --output "${vcf_file}"
    """
}
