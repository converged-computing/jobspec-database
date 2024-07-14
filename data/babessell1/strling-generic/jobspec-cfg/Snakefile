configfile: "config.yaml"

# Define a rule to generate the list of BAM files in the input directory
rule get_bam_files:
    output:
        "bam_files.txt"
    run:
        import os
        with open(output[0], "w") as f:
            bam_files = [f for f in os.listdir(config["BAM_DIR"]) if f.endswith(".bam")]
            f.write("\n".join(bam_files))

# Define a rule to extract STRs for each BAM file
rule extract_str:
    input:
        bam = config["BAM_DIR"] + "{sample}.bam",
        ref = config["REF_FASTA"]
    output:
        "str-results/{sample}.bin"
    shell:
        "{config['STRLING']} extract -f {input.ref} {input.bam} {output}"

# Define a rule to call STRs for each BAM file
rule call_str:
    input:
        bam = config["BAM_DIR"] + "{sample}.bam",
        ref = config["REF_FASTA"],
        bin = "str-results/{sample}.bin"
    output:
        "str-results/{sample}.vcf"
    shell:
        "{config['STRLING']} call --output-prefix str-results/{wildcards.sample} -f {input.ref} {input.bam} {input.bin}"

# Create a rule to create the 'str-results/' directory
rule create_output_dir:
    output:
        directory("str-results")

# Define a workflow to run all the steps
workflow bam_to_str:
    # First, create the output directory
    rule create_output_dir

    # Second, get the list of BAM files in the directory
    rule get_bam_files

    # Third, run the extract_str rule for each BAM file
    rule extract_str

    # Fourth, run the call_str rule for each BAM file
    rule call_str

# Define the 'all' rule to run the entire workflow
rule all:
    input:
        expand("str-results/{sample}.vcf", sample=read_bam_files())
