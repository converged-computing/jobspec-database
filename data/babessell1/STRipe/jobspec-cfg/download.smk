import os

from py.helpers import *
configfile: "config.yaml"

sample_dict = get_sample_dict(config, init=False)

rule_all = []

if config["PROCESS_SHORT"]:
    rule_all.extend([
        expand(os.path.join(config["DATA_DIR"], "short", "{sample}.short{ext}"),
            zip,
            sample=get_samples(sample_dict, "short"),
            num=get_num(sample_dict, "short"),
            ext=get_ext(sample_dict, "short")
        )
    ])
if config["PROCESS_HIFI"]:
    rule_all.extend([
        expand(os.path.join(config["DATA_DIR"], "hifi", "{sample}.hifi{ext}"),
            zip,
            sample=get_samples(sample_dict, "hifi"),
            num=get_num(sample_dict, "hifi"),
            ext=get_ext(sample_dict, "hifi")
        )
    ])
if config["PROCESS_ASSEMBLY"]:
    rule_all.extend([
        expand(os.path.join(config["DATA_DIR"], "assemblies", "{sample}.assembly{ext}"),
            zip,
            sample=get_samples(sample_dict, "assembly"),
            num=get_num(sample_dict, "assembly"),
            ext=get_ext(sample_dict, "assembly")
        )
    ])


rule all:
    input: rule_all


if config["PROCESS_SHORT"]:
    rule download_short:
        output: os.path.join(config["DATA_DIR"], "short", "{sample}.short{ext}")
        params:
            url=lambda wildcards: sample_dict["short"]["url"][wildcards.sample]
        resources:  # 1GB
            mem_mb=1000
        threads: 1
        shell:
            '''
            # if url is not s3 use wget
            mkdir -p raw_data/short_reads
            wget -O "{output}" {params.url}
            '''


if config["PROCESS_HIFI"]:
    rule download_hifi:
        output: os.path.join(config["DATA_DIR"], "hifi", "{sample}.hifi{ext}")
        params:
            url=lambda wildcards: sample_dict["hifi"]["url"][wildcards.sample]
        resources:  # 1GB
            mem_mb=1000
        threads: 1
        shell:
            '''
            # if url is not S3 use wget
            mkdir -p raw_data/short_reads
            if [[ ! "{params.url}" == "https://s3"* ]]; then
                wget -O "{output}" "{params.url}"
            else
                # Convert the URL to S3 format and download using AWS CLI
                s3_key=$(echo "{params.url}" | sed -e 's~https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=~~')
                aws s3 cp "s3://human-pangenomics/${{s3_key}}" "{output}"
            fi
            '''


if config["PROCESS_ASSEMBLY"]:
    rule download_assembly:
        output: os.path.join(config["DATA_DIR"], "assembly", "{sample}.assembly{ext}")
        params:
            url=lambda wildcards: sample_dict["assembly"]["url"][wildcards.sample]
        resources:  # 1GB
            mem_mb=1000
        threads: 1
        shell:
            '''
            # if url is not s3 use wget
            mkdir -p raw_data/assemblies
            wget -O "{output}" {params.url}
            '''