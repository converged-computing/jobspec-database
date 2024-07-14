#!/usr/bin/env nextflow

fpalgo = params.fpalgo ? params.fpalgo.split(",") : []
ssalgo = params.ssalgo ? params.ssalgo.split(",") : []
svalgo = params.svalgo ? params.svalgo.split(",") : []

reffa=file("$params.reffa")
dbsnp=file("$params.dbsnp")
index_path=file("$params.genome")
capturebed = file("$params.capture")
capturedir = file("$params.capturedir")

alignopts = ''
if (params.markdups == 'fgbio_umi' || params.markdups == 'picard_umi') {
   alignopts='-u'
}
repoDir=workflow.projectDir
if (params.repoDir) {
   repoDir=params.repoDir
}
ponopt=''
if (params.pon) {
   ponopt="-q $params.pon"
}

input_dir = file("$params.input")


if (params.startFromBcl) {
  sample_sheet = file("$params.input/SampleSheet.csv")
  if( ! input_dir || ! sample_sheet) { error "Could not find required files" }
} else {
  sample_sheet = Channel.empty()
  if( ! input_dir) { error "Could not find required files" }
}

process mutate_sample_sheet {
  label 'python'
  input:
  file sample_sheet_file from sample_sheet
  output:
  file("NewSampleSheet.csv") into bcl_convert_sample_sheet
  when:
  params.startFromBcl == true
  shell:
    """
    python ${repoDir}/process_scripts/uab/mutateSampleSheet.py ${sample_sheet_file} "${params.overrideCycles}"
    """
}

process bcl_convert {
  label 'bcl_convert'
  publishDir "$params.output/fastqs", mode: 'copy'
  input:
  file input_bcl from input_dir
  file sample_sheet from bcl_convert_sample_sheet
  output:
  file("*.fastq.gz") into dtrim_reads
  file("SampleSheet.csv")
  when:
  params.startFromBcl == true
  script:
  """
  bcl-convert bcl-convert \
    --bcl-input-directory ${input_bcl} --sample-sheet ${sample_sheet} --output-directory ./ --no-lane-splitting true --force
  cp ${sample_sheet} SampleSheet.csv
  rm -rf Undetermined*
  """
}

if(params.startFromBcl) {
  dtrim_reads
    .flatten()
    .map { read -> [read.name.split("_")[0], read] }
    .groupTuple(by: 0)
    .set { reads; }
} else {
  Channel.fromPath(input_dir+"/*.fastq.gz")
    .map { read -> [read.name.split("_")[0], read] }
    .groupTuple(by: 0)
    .set { reads; }
}



process dtrim {
  label 'trim'
  publishDir "$params.output/$sampleid/dnaout", mode: 'copy'
  input:
  set sampleid, file(fqs) from reads
  output:
  set sampleid,file("${sampleid}.trim.R1.fastq.gz"),file("${sampleid}.trim.R2.fastq.gz"),file("${sampleid}.trimreport.txt") into treads
  script:
  """
  bash ${repoDir}/process_scripts/preproc_fastq/trimgalore.sh -f -p ${sampleid} ${fqs}
  """
}
process dalign {
  label 'dnaalign'
  publishDir "$params.output/$sampleid/dnaout", mode: 'copy'
  input:
  set sampleid,file(fq1),file(fq2),file(trimout) from treads
  output:
  set sampleid,file("${sampleid}.bam"),file("${sampleid}.bam.bai") into cnvbam
  set sampleid, file("${sampleid}.bam"),file("${sampleid}.bam.bai"),file(trimout) into qcbam
  set sampleid,file("${sampleid}.bam"), file("${sampleid}.bam.bai") into msibam
  set sampleid,file("${sampleid}.bam"),file("${sampleid}.bam.bai") into align
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory

  bash ${repoDir}/process_scripts/alignment/dnaseqalign.sh -r $index_path -p $sampleid -x ${fq1} -y ${fq2} -c ${task.cpus} -m \${memory} $alignopts
  """
}
process abra2 {
  label 'abra2'
  publishDir "$params.output/$sampleid/dnaout", mode: 'copy'
  input:
  set sampleid,file(sbam),file(bai) from align
  output:
  set sampleid,file("${sampleid}.bam"),file("${sampleid}.bam.bai") into itdbam
  set sampleid,file("${sampleid}.bam"), file("${sampleid}.bam.bai") into svbam
  set sampleid, file("${sampleid}.bam"),file("${sampleid}.bam.bai") into mdupbam
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/alignment/abra2.sh -r $index_path -p $sampleid -b ${sbam} -t ${capturebed} -c ${task.cpus} -m \${memory}
  mv ${sbam} ${sampleid}.ori.bam
  mv ${bai} ${sampleid}.ori.bai
  mv ${sampleid}.abra2.bam  ${sampleid}.bam
  mv ${sampleid}.abra2.bam.bai  ${sampleid}.bam.bai
  """
}
process markdups {
  label 'dnaalign'
  publishDir "$params.output/$sampleid/dnaout", mode: 'copy'

  input:
  set sampleid, file(sbam) from mdupbam
  output:
  set sampleid, file("${sampleid}.consensus.bam"),file("${sampleid}.consensus.bam.bai") into togatkbam
  set sampleid,file("${sampleid}.consensus.bam"),file("${sampleid}.consensus.bam.bai") into alt_vc_bam
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/alignment/markdups.sh -a $params.markdups -b $sbam -p $sampleid -r $index_path -c ${task.cpus} -m \${memory}

  mv ${sampleid}.dedup.bam ${sampleid}.consensus.bam
  mv ${sampleid}.dedup.bam.bai ${sampleid}.consensus.bam.bai
  """
}

process dna_bamqc {
  label 'profiling_qc'
  publishDir "$params.output/$sampleid/dnaout", mode: 'copy'
  input:
  set sampleid, file(gbam),file(idx),file(trimreport) from qcbam
  output:
  file("*fastqc*") into fastqc
  file("${sampleid}*txt") into dalignstats
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/alignment/bamqc.sh -c $capturebed -n dna -r $index_path -b ${gbam} -p $sampleid -e ${params.version} -x ${task.cpus} -y \${memory}
  """
}

process cnv {
  label 'structuralvariant'
  publishDir "$params.output/$sampleid/dnacallset", mode: 'copy'
  input:
  set sampleid,file(sbam),file(sidx) from cnvbam

  output:
  file("${sampleid}.call.cns") into cns
  file("${sampleid}.cns") into cnsori
  file("${sampleid}.cnr") into cnr
  file("${sampleid}.answerplot*") into cnvansplot
  file("${sampleid}.*txt") into cnvtxt
  file("${sampleid}.cnv*pdf") into cnvpdf
  when:
  params.skipCNV == false && params.min == false
  script:
  """
  bash ${repoDir}/process_scripts/variants/cnvkit.sh -r $index_path -b $sbam -p $sampleid -d $capturedir
  """
}

process itdseek {
  label 'structuralvariant'
  publishDir "$params.output/$sampleid/dnacallset", mode: 'copy'
  input:
  set sampleid,file(sbam),file(sidx) from itdbam

  output:
  file("${sampleid}.itdseek_tandemdup.vcf.gz") into itdseekvcf
  when:
  params.skipITDSeek == false && params.min == false
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/variants/svcalling.sh -b $sbam -r $index_path -p $sampleid -l $params.itdbed -a itdseek -g $params.snpeff_vers -z ${task.cpus} -m \${memory} -f
  """
}

process gatkbam {
  label 'variantcalling'
  publishDir "$params.output/$sampleid/dnaout", mode: 'copy'

  input:
  set sampleid, file(sbam),file(idx) from togatkbam
  output:
  set sampleid,file("${sampleid}.final.bam"),file("${sampleid}.final.bam.bai") into pindelbam
  set sampleid,file("${sampleid}.final.bam"),file("${sampleid}.final.bam.bai") into mutectbam
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/variants/gatkrunner.sh -a gatkbam -b $sbam -r $index_path -p $sampleid -c ${task.cpus} -m \${memory}
  """
}

process msi {
  label 'profiling_qc'
  publishDir "$params.output/$sampleid/dnacallset", mode: 'copy'
  input:
  set sampleid,file(ssbam),file(ssidx) from msibam
  output:
  file("${sampleid}*") into msiout
  when:
  params.skipMSI == false && params.min == false
  script:
  """
  bash ${repoDir}/process_scripts/variants/msisensor.sh -r ${index_path} -p $sampleid -b ${sampleid}.bam -c $capturebed
  """
}

process pindel {
  label 'pindel'
  publishDir "$params.output/$sampleid/dnacallset", mode: 'copy'
  input:
  set sampleid,file(ssbam),file(ssidx) from pindelbam
  output:
  file("${sampleid}.pindel_tandemdup.vcf.gz") into tdvcf
  set sampleid,file("${sampleid}.pindel.vcf.gz") into pindelvcf
  file("${sampleid}.pindel.genefusion.txt") into pindelgf
  when:
  params.min == false
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/variants/svcalling.sh -r $index_path -p $sampleid -l $params.itdbed -a pindel -c $capturebed -g $params.snpeff_vers -z ${task.cpus} -m \${memory} -f
  """
}

process sv {
  label 'structuralvariant'
  publishDir "$params.output/$sampleid/dnacallset", mode: 'copy'

  input:
  set sampleid,file(ssbam),file(ssidx) from svbam
  each algo from svalgo
  output:
  set sampleid,file("${sampleid}.${algo}.vcf.gz") into svvcf
  set sampleid,file("${sampleid}.${algo}.sv.vcf.gz") optional true into svsv
  file("${sampleid}.${algo}.genefusion.txt") into svgf
  when:
  params.min == false
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/variants/svcalling.sh -r $index_path -b ${sampleid}.bam -p $sampleid -a ${algo} -g $params.snpeff_vers -z ${task.cpus} -m \${memory} -f
  """
}

process mutect {
  label 'variantcalling'
  publishDir "$params.output/$sampleid/dnacallset", mode: 'copy'

  input:
  set sampleid,file(ssbam),file(ssidx) from mutectbam
  output:
  set sampleid,file("${sampleid}.mutect.vcf.gz") into mutectvcf
  set sampleid,file("${sampleid}.mutect.ori.vcf.gz") into mutectori
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/variants/germline_vc.sh $ponopt -r $index_path -p $sampleid -b $capturebed -a mutect -c ${task.cpus} -m \${memory}
  bash ${repoDir}/process_scripts/variants/uni_norm_annot.sh -g $params.snpeff_vers -r $index_path -p ${sampleid}.mutect -v ${sampleid}.mutect.vcf.gz
  """
}

process alt_vc {
  label 'variantcalling'
  publishDir "$params.output/$sampleid/dnacallset", mode: 'copy'
  input:
  set sampleid,file(gbam),file(gidx) from alt_vc_bam
  each algo from fpalgo
  output:
  set sampleid,file("${sampleid}.${algo}.vcf.gz") into alt_vcf
  set sampleid,file("${sampleid}.${algo}.ori.vcf.gz") into alt_ori
  script:
  """
  memory=\$(echo ${task.memory} | cut -d ' ' -f1)
  echo \$memory
  bash ${repoDir}/process_scripts/variants/germline_vc.sh -r $index_path -p $sampleid -a ${algo} -b $capturebed -c ${task.cpus} -m \${memory}
  bash ${repoDir}/process_scripts/variants/uni_norm_annot.sh -g $params.snpeff_vers -r $index_path -p ${sampleid}.${algo} -v ${sampleid}.${algo}.vcf.gz
  """
}

Channel
  .empty()
  .mix(mutectvcf,pindelvcf,alt_vcf)
  .groupTuple(by:0)
  .set { vcflist}

process integrate {
  label 'variantcalling'
  publishDir "$params.output/$sampleid/dnavcf", mode: 'copy'
  input:
  set sampleid,file(vcf) from vcflist
  output:
  file("${sampleid}.union.vcf.gz") into unionvcf
  script:
  """
  bash ${repoDir}/process_scripts/variants/union.sh -r $index_path -p $sampleid
  #cp ${sampleid}.union.vcf.gz ${sampleid}.dna.vcf.gz
  """
}
