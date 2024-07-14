import pandas as pd
configfile: "config/config.yaml"

ACCS=pd.read_table(config["sras"],header=0,dtype=str)['SRA']

rule all:
    input:
        expand("sample_info/{acc}.json",acc=ACCS)

rule sample_info:
    output:
        "sample_info/init_{acc}.txt"
    shell:
        """
        touch {output}
        """

rule ftp_fetch:
    output:
        json="sample_info/{acc}.json",
        ftp="sample_info/ftp_urls_{acc}.txt"
    params:
        base_url="https://www.ebi.ac.uk/ena/portal/api/"
    log:
        stderr="logs/{acc}.stderr"
    shell:
        """
        ./scripts/ftp_fetch.sh -a {wildcards.acc} -b sample_info/ 2> {log.stderr}
        """

rule json_summarise:
    params:
        inputdir='sample_info',
        outputname='js_summarise_dummy.tsv',
        outputdir='json_summarise'
    output:
        'json_summarise/js_summarise_dummy.tsv'
    shell:
        """
        Rscript ./scripts/json2tsv.R {config[basedir]}{params.inputdir} {params.outputname} {config[basedir]}{params.outputdir}
        """

