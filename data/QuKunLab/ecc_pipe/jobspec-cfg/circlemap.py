configfile: "circlemap_config.yaml"
    
##define params
#print(config["input_file"])
#print(type(config["input_file"]))
name=config["input_file"].keys()
ecc_master_path=config["ecc_master_path"]
output_path=config["output_path"]

## take AA ref
if config['user_ref'] == 'None':
    if config["reference"] == 'mm10':
        reference = ecc_master_path+'/resource/AA/AmpliconArchitect/data_repo/mm10/mm10.fa'
    elif config["reference"] == 'hg38':
        reference = ecc_master_path+'/resource/AA/AmpliconArchitect/data_repo/GRCh38/hg38full.fa'
    else:
        print("please set reference in yaml in ['mm10','hg38']")
else:
    config_ref = config['user_ref']
    reference=ecc_master_path+'/resource/user_add/reference/'+config_ref+'.fa'

threads=config["threads"]

import warnings
warnings.filterwarnings('ignore')

import os
relative_path = os.path.relpath(output_path, ecc_master_path)
output_path = relative_path


def get_bwa_map_input_fastqs(wildcards):
    print(wildcards)
    return config["input_file"][wildcards.sample]

rule all:
    input:
        ##保留文件 这里要用expand 遍历
        expand("{output_path}/01.mapped_reads/{name}.sam",
               output_path=output_path,
               name=name),
        expand("{output_path}/01.mapped_reads/{name}_sort.bam",
               output_path=output_path,
               name=name),
        expand("{output_path}/02.circle_pre/{name}_qname.bam",
               output_path=output_path,
               name=name),
        expand("{output_path}/02.circle_pre/{name}_circle_candidates.bam",
               output_path=output_path,
               name=name),
        expand("{output_path}/02.circle_pre/{name}_sort_circle_candidates.bam",
               output_path=output_path,
               name=name),
        expand("{output_path}/03.circlemap.results/{name}_circlemap_result.bed",
               output_path=output_path,
               name=name),

##01.bwa mapping
rule bwa_map:
    input:
        expand("{reference}", reference=reference),
        get_bwa_map_input_fastqs
        
    output:
        output_path+'/01.mapped_reads/{sample}.sam',
    
    log:
        output_path+'/logs/{sample}.01.log',
            
    threads: config["threads"]
    shell:
        "bwa mem -t {threads} \
		-q \
            {input} > {output}"

##02.samtools_sort
rule samtools_sort:
    input:
        output_path+'/01.mapped_reads/{sample}.sam',
    output:
        output_path+'/01.mapped_reads/{sample}_sort.bam',
    log:
        output_path+'/logs/{sample}.02.log',

    threads: config["threads"]
    shell:
        "samtools sort -@ {threads} -o {output} {input} ; "
        "samtools index -@ {threads} {output}"

##03.circlemap sort candidates
rule circlemap_extra:
    input:
        output_path+'/01.mapped_reads/{sample}.sam',
    output:
        qname=output_path+'/02.circle_pre/{sample}_qname.bam',
        circle=output_path+'/02.circle_pre/{sample}_circle_candidates.bam',
        candidates=output_path+'/02.circle_pre/{sample}_sort_circle_candidates.bam',
    log:
        output_path+'/logs/{sample}.03.log',

    threads: config["threads"]
    shell:
        "samtools sort -@ {threads} -n -o {output.qname} {input} ; "
        "Circle-Map ReadExtractor -o {output.circle} -i {output.qname};"
        "samtools sort -@ {threads} -o {output.candidates} {output.circle} ; "
        "samtools index -@ {threads} {output.candidates}"
        
##04.detecting circle dna
rule circlemap_detect:
    input:
        candidates=output_path+'/02.circle_pre/{sample}_sort_circle_candidates.bam',
        qname=output_path+'/02.circle_pre/{sample}_qname.bam',
        sbam=output_path+'/01.mapped_reads/{sample}_sort.bam',
        fasta=expand("{reference}", reference=reference),
    output:
        output_path+'/03.circlemap.results/{sample}_circlemap_result.bed'
    log:
        output_path+'/logs/{sample}.04.log',

    threads: config["threads"]
    shell:
        "Circle-Map Realign -i {input.candidates} \
            -qbam {input.qname} \
            -sbam {input.sbam} \
            -fasta {input.fasta} \
            -o {output} \
            -t {threads}"
        










