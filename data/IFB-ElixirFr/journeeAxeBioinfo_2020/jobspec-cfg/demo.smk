# add conda: "xx.yml"

SAMPLES, = glob_wildcards(config["dataDir"]+"{sample}.fastq.gz")
BIDX = ["1","2","3","4","rev.1","rev.2"]

rule all:
  input:
    expand("FastQC/{sample}_fastqc.html", sample=SAMPLES),
    expand("Tmp/Otauri.{ext}.bt2", ext=BIDX),
    expand("Tmp/{sample}.sam", sample=SAMPLES),
    expand("Result/{sample}_sort.bam.bai", sample=SAMPLES),
    expand("Result/Counts/{sample}_ftc.txt", sample=SAMPLES),
    "Result/counts_matrix.txt"

rule matrix_counts:
  output:
    "Result/counts_matrix.txt"
  input:
    countfile=expand("Tmp/{sample}_ftc7.txt", sample=SAMPLES),
    geneID=expand("Tmp/{sample}_ftc1.txt", sample=SAMPLES)
  log:
    "Logs/matrix_counts.log"
  shell:
    """cp {input.geneID[0]} Tmp/ftc_geneID.txt ; paste Tmp/ftc_geneID.txt {input.countfile} > {output}"""

rule extract_counts:
  output:
    col7="Tmp/{sample}_ftc7.txt",
    col1="Tmp/{sample}_ftc1.txt"
  input:
    "Result/Counts/{sample}_ftc.txt"
  shell: """cut -f 7 {input} | sed 1d > {output.col7} ; cut -f 1 {input} | sed 1d > {output.col1} """

rule counting:
  output:
    "Result/Counts/{sample}_ftc.txt"
  input:
    bam="Result/{sample}_sort.bam",
    annot=config["dataDir"]+config["annots"]
  params: t="gene", g="ID", s="2"
  log:
    "Logs/{sample}_counts.log"
  conda: "condaEnv4SmkRules/counting.yml"
  shell:
    "featureCounts -t {params.t} -g {params.g} -a {input.annot} -s {params.s} -o {output} {input.bam} &> {log}"

rule sam2bam_sort:
  output:
    bam="Result/{sample}_sort.bam",
    bai="Result/{sample}_sort.bam.bai"
  input:
    "Tmp/{sample}.sam"
  log:
    sort="Logs/{sample}_sam2bam_sort.log",
    index="Logs/{sample}_bam2bai.log"
  conda: "condaEnv4SmkRules/sam2bam_sort.yml"
  shell: 
    "samtools sort -O bam -o {output.bam} {input} 2> {log.sort} ;"
    "samtools index {output.bam} 2> {log.index}"


rule bwt2_mapping:
  output:
    "Tmp/{sample}.sam"
  input:
    config["dataDir"]+"{sample}.fastq.gz",
    expand("Tmp/Otauri.{ext}.bt2", ext=BIDX)
  log:
    "Logs/{sample}_bwt2_mapping.log"
  conda: "condaEnv4SmkRules/bwt2_mapping.yml"
  shell: "bowtie2 -x Tmp/Otauri -U {input[0]} -S {output} 2> {log} "

rule genome_bwt2_index:
  output:
    expand("Tmp/Otauri.{ext}.bt2", ext=BIDX)
  input:
    config["dataDir"]+config["genome"]
  log:
    log1="Logs/genome_bwt2_index.log1",
    log2="Logs/genome_bwt2_index.log2"
  conda: "condaEnv4SmkRules/bwt2_mapping.yml"
  shell: "bowtie2-build {input} Tmp/Otauri 1>{log.log1} 2>{log.log2}"

rule fastqc:
  output: 
    "FastQC/{sample}_fastqc.zip",
    "FastQC/{sample}_fastqc.html"
  input: 
    config["dataDir"]+"{sample}.fastq.gz"
  log:
    log1="Logs/{sample}_fastqc.log1",
    log2="Logs/{sample}_fastqc.log2"
  conda: "condaEnv4SmkRules/fastqc.yml"
  shell: "fastqc --outdir FastQC/ {input} 1>{log.log1} 2>{log.log2}"


