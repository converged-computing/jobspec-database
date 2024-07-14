#!/usr/bin/env python
import os
import pandas as pd
import sys
#### ENVS ####

# Titania (python 3.7.10)
# mb (python 3.9.16)
    # compleasm
	
# call mains from modules folder using shell calls 

# create a list of files in directory mags ending in .fna


mags_list = os.listdir("mags/")
print (mags_list)
rule all:
    input:
        expand("busco_out/{mag}/summary.txt", mag=mags_list),
        expand("eukcc_out/{mag}/", mag=mags_list),
        expand("xgb_out/{mag}.out", mag=mags_list),
        "mag_stats_x.csv"

rule run_busco:
    input: 
        "mags/{mag}"
    output:
        "busco_out/{mag}/summary.txt"
    conda:
        "../envs/mb.yaml"
    threads:
        240
    shell:
        "compleasm run -a {input} -t {threads} -l eukaryota -L resources/mb_downloads/ -o busco_out/{wildcards.mag}"

rule run_eukcc:
    input:
        "mags/{mag}"
    output:
        directory("eukcc_out/{mag}/")
    threads:
        240
    conda:
        "../envs/Titania.yaml"
    shell:
        "eukcc single --out {output} --threads {threads} {input}"

rule run_xgb_class:
    input:
        "mags/{mag}"
    output:
        "xgb_out/{mag}.out"
    conda:
        "../envs/Titania.yaml"
    shell:
        "python resources/4CAC/classify_xgb.py -f {input} -o {output}"

rule mag_stat:
    input:
        "mags/"
    output:
        "mag_stats.csv"
    conda:
        "../envs/Titania.yaml"
    shell:
        "python modules/mag_stats.py -f {input}"

rule mag_stat_b:
    input:
        magstat=rules.mag_stat.output
    output:
        "mag_stats_b.csv"
    conda:
        "../envs/Titania.yaml"
    shell:
        "python modules/busco_parse.py -b busco_out/ -m {input.magstat} -o {output}"


rule mag_stat_e:
    input:
        magstat=rules.mag_stat_b.output
    output:
        "mag_stats_e.csv"
    conda:
        "../envs/Titania.yaml"
    shell:
        "python modules/eukcc_parse.py -e eukcc_out -m {input} -o {output}"
        
rule mag_stat_x:
    input:
        magstat=rules.mag_stat_e.output
    output:
        "mag_stats_x.csv"
    conda:
        "../envs/Titania.yaml"
    shell:
        "python modules/xg_parse.py -x xgb_out -m {input} -o {output}"

rule cleaner:
    input:
       base=rules.mag_stat.output,
       comp=rules.mag_stat_b.output,
       euk=rules.mag_stat_e.output
    output:
        touch("cleaner.done")
    run:
        shell("rm {input.base}")
        shell("rm {input.comp}")
        shell("rm {input.euk}")