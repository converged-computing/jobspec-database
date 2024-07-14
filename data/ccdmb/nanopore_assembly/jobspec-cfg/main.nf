
// before running this script, you need to manually concatenate the demultiplexed fastq files. 
// This script expects one fastq file per genome. guppy can do the barcode demultiplexing during basecalling
// now and creates lots of small fastq (or fastq.gz) files in folders called 'barcodeXX', where
// 'XX' stands for the barcode number i.e. 'barcode06'
// For this script to work and name everything correctly you need to concatenate all those files into one .fastq 
// file, NOT fastq.gz!
// There might be a problem of read duplication. To be sure run this bit of code over the concatenated fastq files
//
// for sample in `ls *.fastq | cut -f1 -d'.'`; do cat $sample.fastq | seqkit rmdup -n -o $sample.clean.fastq; done

def helpMessage() {
    log.info"""
    # Nanopore genome polishing
    A pipeline for polishing genomes assembled from Oxford Nanopore reads using Racon and Medaka.

    ## Examples
    nextflow run nanopore_polishing.nf \
    --nanoporeReads "03-trimmed-fastq/*.fastq.gz"

    ## Parameters
    --nanoporeReads <glob>
        Required
        A glob of the fastq.gz files of the adapter and barcode trimmed reads.
        The basename of the file needs to match the basename of the respective genome.

    --canuSlow
        Default: false
        Disables canu fast mode.

    --outdir <path>
        Default: `assembly`
        The directory to store the results in.

    ## Exit codes
    - 0: All ok.
    - 1: Incomplete parameter inputs.
    """
}

if (params.help) {
    helpMessage()
    exit 0
}

if ( params.nanoporeReads ) {
    nanoporeReads = Channel
    .fromPath(params.nanoporeReads, checkIfExists: true, type: "file")
    .map {file -> [file.simpleName, file]}
    .tap { readsForAssembly }
} else {
    log.info "No nanopore reads supplied, use '--nanoporeReads' and make sure to include '*.fastq.gz'"
    exit 1
}


process canu_version {

    label "canu"

    output:
    path 'versions.txt' into canu_version

    """
    echo canu: >> versions.txt
    canu --version >> versions.txt
    echo --------------- >> versions.txt
    """
}


process minimap2_version {

    label "minimap2"

    output:
    path 'versions.txt' into minimap2_version

    """
    echo minimap2: >> versions.txt
    minimap2 --version >> versions.txt
    echo --------------- >> versions.txt
    """
}


process racon_version {

    label "racon"

    output:
    path 'versions.txt' into racon_version

    """
    echo racon: >> versions.txt
    racon --version >> versions.txt
    echo --------------- >> versions.txt
    """
}


process medaka_version {

    label "medaka"

    output:
    path 'versions.txt' into medaka_version

    """
    echo medaka: >> versions.txt
    medaka --version >> versions.txt
    echo --------------- >> versions.txt
    """
}


process seqkit_version {

    label "seqkit"

    output:
    path 'versions.txt' into seqkit_version

    """
    echo seqkit: >> versions.txt
    seqkit version >> versions.txt
    echo --------------- >> versions.txt
    """
}


process version {

    input:
    path "canu.txt" from canu_version
    path "racon.txt" from racon_version
    path "minimap.txt" from minimap2_version
    path "medaka.txt" from medaka_version
    path "seqkit.txt" from seqkit_version

    publishDir "${params.outdir}/", mode: 'copy', pattern: 'versions.txt'

    output:
    path "versions.txt"

    script:
    """
    cat canu.txt racon.txt medaka.txt minimap.txt seqkit.txt > versions.txt
    """
}


// genome assembly
process canu {

    label "canu"
    tag {sampleID}
    publishDir "${params.outdir}/04-canu-assembly", mode: 'copy', pattern: '*.fasta'
    publishDir "${params.outdir}/04-canu-assembly", mode: 'copy', pattern: '*.fasta.gz'
    publishDir "${params.outdir}/04-canu-assembly", mode: 'copy', pattern: '*.report'

    input:
    tuple sampleID, 'input.fastq.gz' from readsForAssembly

    output:
    tuple sampleID, "${sampleID}.contigs.fasta", 'input.fastq.gz' into minimap2
    tuple sampleID, "${sampleID}.correctedReads.nanopore.fasta.gz" into correctedReads
    path "${sampleID}.canu.nanopore.report"

    script:
    // See: https://groovy-lang.org/operators.html#_elvis_operator
    fast_option = params.canuSlow ? "" : "-fast "

    """
    canu \
    -p ${sampleID} \
    -d ${sampleID} \
    genomeSize=45m \
    minInputCoverage=5 \
    stopOnLowCoverage=5 \
    ${fast_option} \
    -nanopore input.fastq.gz

    cp ${sampleID}/*contigs.fasta ${sampleID}.contigs.fasta
    cp ${sampleID}/*correctedReads.fasta.gz ${sampleID}.correctedReads.nanopore.fasta.gz
    cp ${sampleID}/*.report ${sampleID}.canu.nanopore.report
    """
}


process minimap2 {

    tag {sampleID}
    label "minimap2"

    input:
    tuple sampleID, 'input.fasta', 'input.fastq.gz' from minimap2

    output:
    tuple sampleID, 'input.fasta', 'input.fastq.gz', 'minimap.racon.paf' into racon

    """
    minimap2 \
        input.fasta \
        input.fastq.gz > minimap.racon.paf
    """
}


// polishing step 1
process racon {

    label "racon"
    tag {sampleID}
    publishDir "${params.outdir}/05-racon-polish", mode: 'copy', pattern: '*.fasta'

    input:
    tuple sampleID, 'input.fasta', 'input.fastq.gz', 'minimap.racon.paf' from racon

    output:
    tuple sampleID, "${sampleID}.contigs.racon.fasta", 'input.fastq.gz' into medaka

    """
    racon -m 8 -x -6 -g -8 -w 500 -t 14\
    --no-trimming \
    input.fastq.gz \
    minimap.racon.paf \
    input.fasta > ${sampleID}.contigs.racon.fasta
    """
}


// polishing step 2
process medaka {

    label "medaka"
    tag {sampleID}

    input:
    tuple sampleID, 'input.fasta', 'input.fastq.gz' from medaka

    output:
    tuple sampleID, "${sampleID}.contigs.racon.medaka.fasta", 'input.fastq.gz' into seqkit

    """
    medaka_consensus \
    -d input.fasta \
    -i input.fastq.gz \
    -o ${sampleID}_medaka_output \
    -m r941_min_sup_g507 \
    -t 14

    cp ${sampleID}_medaka_output/consensus.fasta ${sampleID}.contigs.racon.medaka.fasta
    """
}


process seqkit {

    label "seqkit"
    tag {sampleID}

    publishDir "${params.outdir}/06-medaka-polish", mode: 'copy', pattern: '*.fasta'

    input:
    tuple sampleID, "input.fasta", 'input.fastq.gz' from seqkit

    output:
    tuple sampleID, "${sampleID}.contigs.racon.medaka.fasta", 'input.fastq.gz'

    """
    seqkit sort -lr input.fasta > ${sampleID}.fasta
    seqkit replace -p '.+' -r '${sampleID}_ctg_{nr}' --nr-width 2 ${sampleID}.fasta > ${sampleID}.contigs.racon.medaka.fasta
    """
}
