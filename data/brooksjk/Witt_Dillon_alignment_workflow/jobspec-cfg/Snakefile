configfile: "config.yaml"

ancient_patterns = [
    config["SAMPLES"] + "/Ancient_{ancient}_1.fastq.gz",
    config["SAMPLES"] + "/Ancient_{ancient}_2.fastq.gz"
]

modern_patterns = [
    config["SAMPLES"] + "/Modern_{modern}_1.fastq.gz",
    config["SAMPLES"] + "/Modern_{modern}_2.fastq.gz"
]

se_ancient_patterns = config["SAMPLES"] + "/SE_Ancient_{se}.fastq.gz"

ancient_samples = set()
modern_samples = set()
se_ancient_samples = set()

for pattern in ancient_patterns:
    ancient_samples.update(glob_wildcards(pattern).ancient)

for pattern in modern_patterns:
    modern_samples.update(glob_wildcards(pattern).modern)

se_ancient_samples.update(glob_wildcards(se_ancient_patterns).se)

ancient_samples = sorted(ancient_samples)
modern_samples = sorted(modern_samples)
se_ancient_samples = sorted(se_ancient_samples)

ANCIENT_UNIQ = list(ancient_samples)
MODERN_UNIQ = list(modern_samples)
SE_UNIQ = list(se_ancient_samples)

print(SE_UNIQ) 

rule all:
    input:
        o1=expand(config["OUTPUT"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb.dup.bam", ANCIENT_UNIQ=ANCIENT_UNIQ),
        o2=expand(config["OUTPUT"] + "/Modern_{MODERN_UNIQ}_bwa_wb.dup.bam", MODERN_UNIQ=MODERN_UNIQ),
        o3=expand(config["OUTPUT"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.dup.bam", SE_UNIQ=SE_UNIQ)

#ANCIENT SAMPLES

rule ancient_trim:
    input:
        software=config["ADAPTREMOVE"],
        ref=config["REF"],
        I1=config["SAMPLES"] + "/Ancient_{ANCIENT_UNIQ}_1.fastq.gz",
        I2=config["SAMPLES"] + "/Ancient_{ANCIENT_UNIQ}_2.fastq.gz"
    output:
        pair1=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.pair1.truncated"),
        pair2=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.pair2.truncated"),
        O3=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.collapsed"),
        O4=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.collapsed.truncated"),
        O5=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.discarded"),
        O6=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.settings"),
        O7=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.singleton.truncated")
    params:
        reads=config["SAMPLES"] + "/Ancient_{ANCIENT_UNIQ}",
        sample="Ancient_{ANCIENT_UNIQ}"
    resources: cpus=24,mem_mb=40000,time_min=10080
    shell:
        """
            {input.software} --collapse --trimns --trimqualities --mm 5 --minlength 25 --file1 {input.I1} --file2 {input.I2} --basename {params.sample}_trimmed --threads 24
        """

rule ancient_bwa_aln_1:
    input:
        ref=config["REF"],
        pair1=config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.pair1.truncated"
    output:
        sai1=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb_1.sai")
    resources: cpus=24,mem_mb=40000,time_min=10080
    shell:
        """
            module load bwa
            bwa aln -l 1024 -n 0.01 -o 2 -t 24 {input.ref} {input.pair1} > {output.sai1}
        """

rule ancient_bwa_aln_2:
    input:
        ref=config["REF"],
        pair2=config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.pair2.truncated"
    output:
        sai2=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb_2.sai")
    resources: cpus=24,mem_mb=40000,time_min=10080
    shell:
        """
            module load bwa
            bwa aln -l 1024 -n 0.01 -o 2 -t 24 {input.ref} {input.pair2} > {output.sai2}
        """
rule ancient_bwa_sampe:
    input:
        ref=config["REF"],
        pair1=config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.pair1.truncated",
        pair2=config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_trimmed.pair2.truncated",
        sai1=config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb_1.sai",
        sai2=config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb_2.sai"
    output:
        bam=temp(config["WORKDIR"]  + "/Ancient_{ANCIENT_UNIQ}_bwa_wb.bam")
    resources: cpus=1, mem_mb=32000, time_min=1440
    shell:
        """
            module load bwa
            module load samtools
            bwa sampe {input.ref} {input.sai1} {input.sai2} {input.pair1} {input.pair2} | samtools view -b -S -q 25 -F 4 > {output.bam}
        """
        
rule ancient_samtools:
    input:
        bam=config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb.bam",
    output:
        sorted_bam=temp(config["WORKDIR"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb.srt.bam"),
        dedup_bam=config["OUTPUT"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb.dup.bam",
        dedup_bam_index=config["OUTPUT"] + "/Ancient_{ANCIENT_UNIQ}_bwa_wb.dup.bam.bai"
    resources: cpus=1, mem_mb=32000, time_min=1440
    shell:
        """
            module load samtools
            samtools sort {input.bam} -o {output.sorted_bam}
            samtools rmdup -s {output.sorted_bam} {output.dedup_bam}
            samtools index {output.dedup_bam} {output.dedup_bam_index}
        """
        
#MODERN SAMPLES


rule modern_trim:
    input:
        software=config["ADAPTREMOVE"],
        ref=config["REF"],
        I1=config["SAMPLES"] + "/Modern_{MODERN_UNIQ}_1.fastq.gz",
        I2=config["SAMPLES"] + "/Modern_{MODERN_UNIQ}_2.fastq.gz"
    output:
        pair1=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.pair1.truncated"),
        pair2=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.pair2.truncated"),
        O3=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.collapsed"),
        O4=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.collapsed.truncated"),
        O5=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.discarded"),
        O6=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.settings"),
        O7=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.singleton.truncated")
    params:
        reads=config["SAMPLES"] + "/{MODERN_UNIQ}",
        sample="Modern_{MODERN_UNIQ}"
    resources: cpus=40,mem_mb=40000,time_min=10080
    shell:
        """
            {input.software} --collapse --trimns --trimqualities --mm 5 --minlength 25 --file1 {input.I1} --file2 {input.I2} --basename {params.sample}_trimmed --threads 24
        """

rule modern_bwa_mem:
    input:
        ref=config["REF"],
        pair1=config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.pair1.truncated",
        pair2=config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_trimmed.pair2.truncated"
    output:
        sam=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}__bwa_wb.sam")
    resources: cpus=24,mem_mb=40000,time_min=10080
    shell:
        """
            module load bwa
            bwa mem -t 24 {input.ref} {input.pair1} {input.pair2} > {output.sam}
        """
        
rule modern_samtools:
    input:
        sam=config["WORKDIR"]+"/Modern_{MODERN_UNIQ}__bwa_wb.sam"
    output:
        bam=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_bwa_wb.bam"),
        sorted_bam=temp(config["WORKDIR"] + "/Modern_{MODERN_UNIQ}_bwa_wb.srt.bam"),
        dedup_bam=config["OUTPUT"] + "/Modern_{MODERN_UNIQ}_bwa_wb.dup.bam",
        dedup_bam_index=config["OUTPUT"] + "/Modern_{MODERN_UNIQ}_bwa_wb.dup.bam.bai"
    resources: cpus=1, mem_mb=32000, time_min=1440
    shell:
        """
            module load samtools
            samtools view -b -q 30 -F 4 {input.sam} > {output.bam}
            samtools sort {output.bam} -o {output.sorted_bam}
            samtools rmdup -s {output.sorted_bam} {output.dedup_bam}
            samtools index {output.dedup_bam} {output.dedup_bam_index}
        """
        
#SE SAMPLES

rule se_ancient_trim:
    input:
        software=config["ADAPTREMOVE"],
        ref=config["REF"],
        I1=config["SAMPLES"] + "/SE_Ancient_{SE_UNIQ}.fastq.gz"
    output:
        pair1=temp(config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_trimmed.truncated"),
        O5=temp(config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_trimmed.discarded"),
        O6=temp(config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_trimmed.settings")
    params:
        reads=config["SAMPLES"] + "/SE_Ancient_{SE_UNIQ}",
        sample="SE_Ancient_{SE_UNIQ}"
    resources: cpus=24,mem_mb=40000,time_min=10080
    shell:
        """
            {input.software} --file1 {input.I1} --basename {params.sample}_trimmed --threads 24
        """

rule se_ancient_bwa_aln:
    input:
        ref=config["REF"],
        pair1=config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_trimmed.truncated"
    output:
        sai=temp(config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.sai")
    resources: cpus=24,mem_mb=40000,time_min=10080
    shell:
        """
            module load bwa
            bwa aln -l 1024 -n 0.01 -o 2 -t 24 {input.ref} {input.pair1} > {output.sai}
        """

rule se_ancient_bwa_sampe:
    input:
        ref=config["REF"],
        pair1=config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_trimmed.truncated",
        sai=config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.sai"
    output:
        bam=temp(config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.bam"),
    resources: cpus=1, mem_mb=32000, time_min=1440
    shell:
        """
            module load bwa
            module load samtools
            bwa samse {input.ref} {input.sai} {input.pair1} | samtools view -b -S -q 25 -F 4 > {output.bam}
        """
        
rule se_ancient_samtools:
    input:
        bam=config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.bam",
    output:
        sorted_bam=temp(config["WORKDIR"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.srt.bam"),
        dedup_bam=config["OUTPUT"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.dup.bam",
        dedup_bam_index=config["OUTPUT"] + "/SE_Ancient_{SE_UNIQ}_bwa_wb.dup.bam.bai"
    resources: cpus=1, mem_mb=32000, time_min=1440
    shell:
        """
            module load samtools
            samtools sort {input.bam} -o {output.sorted_bam}
            samtools rmdup -s {output.sorted_bam} {output.dedup_bam}
            samtools index {output.dedup_bam} {output.dedup_bam_index}
        """