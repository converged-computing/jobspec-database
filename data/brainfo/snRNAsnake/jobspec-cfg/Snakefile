configfile: "config.yaml"

rule all:
    input:
        h5=expand("{h5dir}/{project_name}_ad_all.h5ad", h5dir=config["project"]['h5dir'], project_name=config["project"]["project_name"]),
        cell_count=expand("{datadir}/{project_name}_sample_nuclei_counts.tsv", datadir=config["project"]["datadir"], project_name=config["project"]["project_name"]),
        meta=expand("{datadir}/{project_name}_ad_all.obs.tsv", datadir=config["project"]["datadir"], project_name=config["project"]["project_name"])

rule simple_clean:
    input:
        info=config["files"]["info"]
    output:
        h5=expand("{h5dir}/{project_name}_ad_all.h5ad", h5dir=config["project"]['h5dir'], project_name=config["project"]["project_name"]),
        cell_count=expand("{datadir}/{project_name}_sample_nuclei_counts.tsv", datadir=config["project"]["datadir"], project_name=config["project"]["project_name"]),
        meta=expand("{datadir}/{project_name}_ad_all.obs.tsv", datadir=config["project"]["datadir"], project_name=config["project"]["project_name"])
    conda:
        "envs/snRNAsnake_environment.yml"
    script:
        "scripts/qc.py"