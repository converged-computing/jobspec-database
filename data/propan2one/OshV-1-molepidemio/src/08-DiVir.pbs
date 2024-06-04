#!/usr/bin/env bash
#PBS -q mpi
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=28:mem=115g

## DESCRIPTION ##
# Metagenomics analysis targeted viral reads

## USAGE ##
# qsub -v "name=, outdir=, reads1=, reads2=" 08-DiVir.pbs

## Related program ##
#print_comment.py (HemoVir)

###############################################################################
#
# FONCTION USED
#
###############################################################################

function show_help {
    echo "Usage: $0 [options] -i input.fastq.gz -o output_dir -f host_genome.fa -l 200 -r virus_ref.fa -p 2 -t 4 -m 32

    Options:

    -h              Show this help message and exit
    -i input        Input fastq.gz file from Minion sequencing
    -o dir          Path of the output directory
    -f file         Fasta file form host (to sort against the host)
    -l #            min_length of long reads (default=200nt)
    -r file         Fasta file known virus
    -p #            Nombre de polishing à faire (default = 1)
    -t #            Number of threads
    -m #            Memory used (Gb)

Example:
$0 -i input.fastq.gz -o output_dir -f host_genome.fa -r oshv.fa -l 200 -p 2 -t 4 -m 32" 
}

# Fonction to align reads with contigs
Alignment_bowtie()
{ 
    # Controle parameter
    if [ $# -eq 4 ]; then
        echo "Alignment_bowtie function with $NCPUS CPUs"
        echo -e "genome=$1"
        echo -e "R1=$2"
        echo -e "R2=$3"
        echo -e "bamfile=$4"
        #source activate bowtie2
    else
        echo -e "There are $# arguments instead of 4"
    fi
    local genome=$(basename $1)
    local R1=$2
    local R2=$3
    local bamfile=$(basename $R1 .fastq.gz).bam
    sortBamFile=$4
    ln -s $1 .
    bowtie2-build -f ${genome} \
        ${genome} \
        >> $logfile \
        2>> $logfile
    # align
    bowtie2 \
        -p $NCPUS \
        -x ${genome} \
        -1 ${R1} \
        -2 ${R2} | \
        samtools view -b > $bamfile \
            2>> $logfile
    # Sort
    samtools sort -T $bamfile \
        -o ${sortBamFile} \
        $bamfile \
        --threads $NCPUS \
        2>> $logfile
    # Index
    samtools index \
        -b ${sortBamFile} \
        2>> $logfile
    #source deactivate
}

# Structure of chunk announcement - (check_start_chunk - code - check_end_chunk)n
# announcement see l181
announcement () { python2 ${SOFT}/print_comment.py "$1" "#" >> ${logfile}; echo -e "\n## ${1}\n" >> $report ; }

check_start_chunk ()
{
    echo -e "$1 start at: $(date +%Y-%m-%d_%Hh%Mm%Ss)" >> $report
    echo -e "$1 start at: $(date +%Y-%m-%d_%Hh%Mm%Ss)" >> $logfile
    cd $output
    # env
    source activate ${path_conda_env}/$1
}

check_end_chunk ()
{
    echo -e "$1 End at: $(date +%Y-%m-%d_%Hh%Mm%Ss)" >> $report
    echo -e "$1 End at: $(date +%Y-%m-%d_%Hh%Mm%Ss)" >> $logfile
    # env
    conda deactivate
    cd $output
}

###############################################################################
#
#SET UP CONFIGURATION VARIABLES & VALIDATE INPUTS
#
###############################################################################

export PATH=/appli/anaconda/3.7/bin/:$PATH

if [ ! -d "$outdir" ]; then
    echo -e "Error 'OUTDIR' dosen't exist, please make the output path first\n"
    show_help
    exit 0
else
    echo "Analysis using DiVir pipeline !"
    cd $outdir
fi

# A defaut config file should be better..
output=${outdir}/${name}
if [ -z ${reads1} ]; then echo -e "\nError 'READS1' is a mandatory argument\n" ; show_help ; exit 0 ; fi
if [ -z ${reads2} ]; then echo "Single end analysis" ; reads2="" ; fi
if [ -z ${name} ]; then echo "name defaut parameter : basename reads1" ; name=$(basename ${reads1%.fastq.gz}) ; fi
if [ -z ${host_genome} ]; then echo "host_genome defaut parameter : Oyster" ;
    host_genome="/home1/datawork/jdelmott/data_jean/oyster.v9.fa" ; fi
if [ -z ${viral_fasta} ]; then echo "viral_fasta defaut parameter : OsHV" ; 
    viral_fasta=/home1/datawork/jdelmott/data_jean/OsHV-1_strain_microVar_variant_A.fasta ; fi
if [ -z ${viral_kraken_db} ]; then echo "viral_kraken_db defaut parameter : /home/ref-bioinfo/tools_data/krakenuniq/DBDIR_viral" \
        ; viral_kraken_db=/home/ref-bioinfo/tools_data/krakenuniq/DBDIR_viral ; fi
if [ -z ${insersize} ]; then echo "insersize defaut parameter: 417" ;
    insersize=417 ; fi
if [ -z ${viral_target_proteins} ]; then echo "viral_target_proteins defaut parameter : OsHV-1" ; 
    viral_target_proteins="/home1/datawork/jdelmott/data_jean/Ostreid_herpesvirus_BDD_prot_V2.fna" ; fi
if [ -z ${viral_genome} ]; then echo "viral_genome and viral_input absent in defaut parameter" ; 
    viral_genome="" ; viral_input=NR_genome_$name ; else viral_input=$(basename $viral_genome .fasta) ; echo -e "viral_input=$(basename $viral_genome .fasta)" ;
fi # faudra mettre la sortie d'assemblage par defaut

if [ -z ${viral_gff} ]; then echo "viral_gff absent in defaut parameter: Prokka output will be used for GFF file" ; 
    viral_gff=${output}/W09-Annotation_Prokka/$viral_input.gff ; fi # test en mettant le path prokka ########################################

if [ -z ${viral_gb} ]; then echo "viral_gff absent in defaut parameter: Prokka output will be used for GBK file" ; 
    viral_gb=${output}/W09-Annotation_Prokka/$viral_input.gbk ; fi # test en mettant le path prokka ########################################

# variable depend of HPC
if [ -z ${max_memory} ]; then echo "max_memory defaut parameter : 110" ; 
    max_memory=110 ; fi
if [ -z ${NCPUS} ]; then echo "max_memory defaut parameter: 28" ; 
    NCPUS=28 ; fi
if [ -z ${path_conda_env} ]; then echo "path_conda_env defaut parameter: /home1/datahome/jdelmott/conda-env" ; 
    path_conda_env="/home1/datahome/jdelmott/conda-env" ; fi
# Pretty modified from METAWRAP pipeline
if [ -z ${SOFT} ]; then echo "SOFT defaut parameter: /home1/datahome/jdelmott" ;
    SOFT=/home1/datahome/jdelmott ; fi
if [ -z $adapter_trimming ]; then echo "adapter_trimming defaut parameter : /home1/datahome/jdelmott/all_adapter.fa" \
        ; adapter_trimming=/home1/datawork/jdelmott/raw_Hemovir/adapter_Hemovir.fa ; fi

# Analysis selection
if [ -z ${skip_assembly} ]; then echo "skip_assembly defaut parameter : no" \
        ; skip_assembly="no" ; fi

# Output directory
if [ ! -d "$name" ];
then
    mkdir $name
    cd $name
    echo "Start DiVir pipeline analysis on $name"
else
    cd $name
fi

###############################################################################
#
#SET UP GLOBAL VARIABLES
#
###############################################################################

#output=${outdir}/${name}
basefile=$(basename $reads1 .fastq.gz)
logfile=$output/${name}_logfile.txt
report=$output/${name}_$(date +%Y-%m-%d_%Hh%Mm)_report.md
if [ ! -d $output/output ]; then mkdir $output/output ; fi
fileReads_host=$output/output/${basefile}_stat_host.csv
fileReads_viral_fasta=$output/output/${basefile}_stat_virus.csv
fileReads_viral_genome=$output/output/${basefile}_stat_virus_asm.csv
if [ $reads2 != "" ]; then basefileReads2=$(basename $reads2 .fastq.gz) ; fi

fastQC=${output}/01-FastQC
fastQC_out1=$fastQC/${basefile}_fastqc.html
fastQC_out2=$fastQC/${basefile}_trim_fastqc.html
trimmomatic=$output/02-Trimmomatic
readsTrim1=$trimmomatic/${basefile}_trim.fastq.gz
if [ $reads2 != "" ]; then readsTrim2=$trimmomatic/${basefileReads2}_trim.fastq.gz ; fi

classification=$output/"03-Reads_classification"

# 
prokka_out=${output}/W09-Annotation_Prokka #### Quand ça sera renommé, changé la l149 

# Les annotations X correspndent à un manquement dans l'énumération des dossiers
Viral_genome_aln=${output}/X01-Aln_asm
Viral_genome_aln_out=$Viral_genome_aln/${basefile}_viral_genome.bam

# Rm duplicate
rm_duplicat=${output}/X02-Remove_duplicat
rm_duplicat_out=$rm_duplicat/${basefile}_viral_genome_noReplicates.bam

pysamstat=${output}/X03-Pysamstat
pysamstat_out=$pysamstat/${name}_raw_count.csv

Variant_calling=${output}/X04-Variant_calling
Variant_calling_out=$Variant_calling/${name}.vcf.gz

annotation_VC=${output}/X05-VC_annotation
annotation_VC_out=$annotation_VC/${name}_annotated.vcf

VC_table=${output}/X06-VC_table
VC_table_SnpSift=$VC_table/${name}_ann_SnpSift.csv
VC_table_bcftools=$VC_table/${name}_ann_bcftools.csv

###############################################################################
#
# Pipeline
#
###############################################################################

echo -e "Commande: ${0}\nStart at: $(date +%Y-%m-%d_%Hh%Mm%Ss)\nInput:\n" >> $logfile

echo -e "output=$output
basefile=$basefile
reads1=$reads1
reads2=$reads2
name=$name
host_genome=$host_genome
viral_fasta=$viral_fasta
viral_kraken_db=$viral_kraken_db
insersize=$insersize
viral_target_proteins=$viral_target_proteins
viral_genome=$viral_genome
viral_gff=$viral_gff
max_memory=$max_memory
NCPUS=$NCPUS
path_conda_env=$path_conda_env
adapter_trimming=$adapter_trimming
skip_assembly=$skip_assembly
viral_input=$viral_input" >> $logfile

echo -e "# Report of DiVir analysis\n\nStart at: $(date +%Y-%m-%d_%Hh%Mm%Ss)\n" > $report

if [ ! -f "$fastQC_out1" ]; then
    announcement "FastQC 1 Control quality of raw reads"
    check_start_chunk "fastqc"
    if [ $reads2 == "" ]; then
        echo -e "\nFirst quality controle on ${reads1} (single ends)... \n" >> $logfile
        fastqc $reads1
    else
        mkdir $fastQC
        echo -e "### First quality controle" >> $report
        echo -e "\nFastQC on ${reads1} and ${reads2} (paired ends)
        More information to interpret the results can be found [here](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/)" \
            >> $report
        fastqc $reads1 $reads2 \
        -o $fastQC \
        -t $NCPUS \
        2>> $logfile
        check_end_chunk "fastqc"
        check_start_chunk "seqkit"
        echo -e "### Metric for raw reads" >> $report
        seqkit stat ${reads1} ${reads2} >> $report
        check_end_chunk "seqkit"
    fi
else
    echo -e "\nSkipped First Read quality control" >> $logfile
fi

if [ ! -f "$readsTrim1" ]; then
    announcement "Trimming reads with trimomatic"
    check_start_chunk "trimmomatic"
    if [ $reads2 == "" ]; then
        #mkdir $trimmomatic
        echo -e "\n Trimming ${reads1}..." >> $logfile
        #trimmomatic PE -phred33 -threads $NCPUS ${reads1} \
        #${reads2} \
        #${readsTrim1} \
        #$(basename $reads1 .fastq.gz)_1unpaired.fastq.gz \
        #ILLUMINACLIP:/home1/datahome/jdelmott/all_adapter.fa:2:30:10 \
        #LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:50
        #echo -e "Reads trim in : ${reads1%.fastq.gz}_trim.fastq.gz \n " >> $logfile
    else
        mkdir $trimmomatic
        cd 02-trimmomatic
        echo -e "\n### Perform Triming of reads" >> $report
        echo -e "\nTrimmomatic on ${reads1} and ${reads2} (paired ends)" >> $report
        echo -e "Parameters are :
        Remove leading low quality or N bases (below quality 3) (LEADING:3)
        Remove trailing low quality or N bases (below quality 3) (TRAILING:3)
        Scan the read with a 4-base wide sliding window, cutting when the average quality per base drops below 15 (SLIDINGWINDOW:4:15)
        Drop reads below the 50 bases long (MINLEN:50)
        source: [trimmomatic](http://www.usadellab.org/cms/?page=trimmomatic)" >> $report 
        if [ ! -f $adapter_trimming ]; then 
            echo "Error : ${adapter_trimming} dosen't exit,
            Verifiy if the file exist or specified using 'adapter_trimming' parameter" \
                >> $logfile
            exit 0
        fi
        trimmomatic PE -phred33 -threads $NCPUS \
            ${reads1} \
            ${reads2} \
            ${readsTrim1} \
            $(basename $reads1 .fastq.gz)_1unpaired.fastq.gz \
            ${readsTrim2} \
            $(basename $reads2 .fastq.gz)_2unpaired.fastq.gz \
            ILLUMINACLIP:$adapter_trimming:2:30:10 \
            LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:50 \
            >> $report \
            2>> $logfile
        echo -e "Trimming reads are $readsTrim1 & $readsTrim2" >> $report
        check_end_chunk "trimmomatic"
        check_start_chunk "seqkit"
        echo -e "\n#### Metric for raw reads\n" >> $report
        seqkit stat ${readsTrim1} ${readsTrim2} >> $report
        check_end_chunk "seqkit"
    fi
else
    echo -e "\nSkipped Trimmomatic trimming" >> $logfile
fi    

if [ ! -f "$fastQC_out2" ]; then
    announcement "FastQC 2 Control quality of raw reads"
    check_start_chunk "fastqc"
    if [ $reads2 == "" ]; then
        echo -e "\nSecond quality controle on ${readsTrim1} (single ends)... \n" >> $logfile
        fastqc $readsTrim1
    else
        mkdir $fastQC
        echo -e "\n### Second quality controle on trimmed reads" >> $report
        echo -e "\nFastQC on ${readsTrim1} and ${readsTrim2} (paired ends)
        More information to interpret the results can be found [here](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/)" \
            >> $report
        fastqc $readsTrim1 $readsTrim2 \
        -o $fastQC \
        -t $NCPUS \
        2>> $logfile
        check_end_chunk "fastqc"
        check_start_chunk "seqkit"
        echo -e "\n### Metric for raw reads\n" >> $report
        seqkit stat ${readsTrim1} ${readsTrim2} >> $report
        check_end_chunk "seqkit"
    fi
else
    echo -e "\nSkipped First Read quality control" >> $logfile
fi

# Reads_classification

# Host_align

#if [ -z ${viral_genome} ]; then 
#    echo "viral_genome defaut parameter assembly contigs"
#    viral_genome="/home1/datawork/jdelmott/data_jean/Ostreid_herpesvirus.faa"
#fi

# Annotation of viral genome
if [ ! -f "$viral_gff" ] ; then
    announcement "Annotated genome previously asm with Prokka"
    check_start_chunk "prokka"
    echo -e "Parameters are :
    Annotation mode -> Viruses
    Compliant -> Force Genbank/ENA/DDJB compliance
    rfam -> Enable searching for ncRNAs with Infernal+Rfam
    protein -> custom special database : $viral_target_proteins" >> $report
    prokka \
        --kingdom Viruses \
        --outdir $(basename $prokka_out) \
        --cpus $NCPUS \
        --locustag ORFi \
        --centre OsHV \
        --rfam \
        --prefix $viral_input \
        $viral_genome \
        --proteins $viral_target_proteins \
            2>> $logfile
    if [ -z ${viral_gff} ]; then echo -e "\nAdding newly GFF to output directory" ; 
        cp $viral_gff $output/output ; fi # OUTPUT
    if [ ! -f "$output/output/viral_target_proteins_correspondance.txt" ] ; then
        echo -e "\nCreat a file to allow the correspondance of predicted protein and Known protein form host
        in $output/output/viral_target_proteins_correspondance.txt" \
            >> $report
        grep -e ">" $viral_target_proteins > $output/output/viral_target_proteins_correspondance.txt
    fi
    check_end_chunk "prokka"
    check_start_chunk "agat" ; cd $prokka_out
    agat_sp_to_tabulated.pl -gff $viral_input.gff -o ${viral_input}_gff.tab \
        2>> $logfile
    cp ${viral_input}_gff.tab $output/output # OUTPUT
    check_end_chunk "agat"
else
    if [ $viral_gff == $viral_input.gff ]; then 
        echo -e "Skip annotation, GFF annotation on $viral_genome already performed" \
            >> $logfile
    else
        echo -e "Skip annotation, GFF: $viral_gff have been provided by user " \
            >> $logfile
    fi  
fi

# Classical virus align
if [ ! -f "$Viral_genome_aln_out" ]; then
    announcement "Aligning trimmed reads with bowtie2 against viral_genome"
    check_start_chunk "bowtie2"
    mkdir $Viral_genome_aln ; cd $Viral_genome_aln
    echo -e "\n$readsTrim1 and $readsTrim2 are align against $viral_genome" >> $report
    Alignment_bowtie "$viral_genome" \
        "$readsTrim1" \
        "$readsTrim2" \
        "${Viral_genome_aln_out%.bam}_raw.bam"
    BAMFILE=${Viral_genome_aln_out%.bam}_raw.bam
    echo '### Metric for raw reads' >> $report
    echo '```' >> $report
    samtools flagstat $BAMFILE >> $report
    echo '```' >> $report
    echo "Reads with MAPQ smaller than INT [20] are skip, Mapping quality reflects the probability that the read is placed incorrectly.
    Thus 20 is 1/100 or 1% chance that the read is incorrectly mapped.
    By setting this parameter to ≥20, we will keep all reads that have 1% or less probability of being mapped incorrec" >> $report
    echo "Only output alignments with number of CIGAR bases consuming query sequence ≥ INT [20] " >> $report 
    samtools view -bq 20 -m 20 $BAMFILE > ${Viral_genome_aln_out%.bam}_q20.bam 2>> $logfile
    samtools sort -T ${Viral_genome_aln_out%.bam}_q20.bam -o $Viral_genome_aln_out ${Viral_genome_aln_out%.bam}_q20.bam \
        2>> $logfile
    samtools index -b $Viral_genome_aln_out 2>> $logfile
    echo '```' >> $report
    samtools flagstat ${Viral_genome_aln_out} >> $report
    echo '```' >> $report
    check_end_chunk "bowtie2"
else
    echo -e "\nAlignment with " >> $logfile
fi

# remove duplicate | Waring syntax old version !
if [ ! -f $rm_duplicat_out ]; then
    announcement "Remove duplicate with Picard Markduplicate"
    check_start_chunk "picard"
    mkdir $rm_duplicat ; cd $rm_duplicat
    echo -e "\nRemoving duplicat of $Viral_genome_aln_out" \
        >> $logfile
    picard MarkDuplicates \
    I=$Viral_genome_aln_out \
    O=${rm_duplicat_out%.bam}_unsort.bam \
    M=${basefile}_MarkDuplicaye_metrics.txt \
    TMP_DIR=$rm_duplicat/${name}_tmp_java \
    #-VALIDATION_STRINGENCY LENIENT \
    REMOVE_DUPLICATES=true \
        2>> $logfile
    samtools sort -T ${rm_duplicat_out%.bam}_unsort.bam \
        -o $rm_duplicat_out \
        ${rm_duplicat_out%.bam}_unsort.bam 2>> $logfile
    samtools index -b $rm_duplicat_out 
        2>> $logfile
    echo -e '\n### Metric After remove Duplicates\n' >> $report
    echo '```' >> $report
    samtools flagstat $rm_duplicat_out >> $report
    echo '```' >> $report
    echo -e ${name} "\t" \
        `samtools view -c $rm_duplicat_out` \
        "\t" \
        `samtools view -c -f4 $rm_duplicat_out` \
        "\t" \
        `samtools view -c -F4 $rm_duplicat_out` \
    > $fileReads_viral_genome 2>> $logfile
    cp $fileReads_viral_genome $output/output # OUTPUT
    check_end_chunk "picard"
else
    echo -e "\nSkipped Removing duplicat" >> $logfile
fi

#cp $fileReads_host $output/output # OUTPUT
#cp $fileReads_viral_fasta $output/output # OUTPUT

if [ ! -f "$pysamstat_out" ]; then
    announcement "Raw count of nucleotides with Pysamstat"
    check_start_chunk "pysamstats"
    echo -e "\nThese datas allow to plot the genome coverage" >> $report
    mkdir $pysamstat ; cd $pysamstat
    if [ ! -f "$viral_genome.fai"]; then
        check_start_chunk "seqkit"
        echo -e "\nIndex $viral_genome" >> $logfile
        seqkit faidx $viral_genome 2>> $logfile
        check_end_chunk "seqkit"
    fi
    pysamstats -f $viral_genome -t variation $rm_duplicat_out \
        --fields=chrom,pos,ref,matches,mismatches,A,C,G,T > $pysamstat_out \
        2>> $logfile
    cp $pysamstat_out $output/output # OUTPUT
    check_end_chunk "pysamstats"
else
    echo -e "\nSkip raw count using pysamstats" >> $logfile
fi

if [ ! -f "$Variant_calling_out" ]; then
    announcement "Variant calling with FreeBayes"
    if [ ! -f "$viral_genome.fai"]; then
        check_start_chunk "seqkit"
        echo -e "\nIndex $viral_genome" >> $logfile
        seqkit faidx $viral_genome 2>> $logfile
        check_end_chunk "seqkit"
    fi
    check_start_chunk "freebayes"
    echo -e "Parameters are :
    -f : FASTA file used to pileup the reads is $viral_genome
    -b : BAM file with the alignement is $rm_duplicat_out
    --use-mapping-quality -> Use mapping quality of alleles when calculating data likelihoods.
    --min-repeat-entropy -> To detect interrupted repeats, build across sequence until it has entropy > 1 bits per bp
    --gvcf -> Write gVCF output, which indicates coverage in uncalled regions.
    --haplotype-length **0** -> Naive variant calling: simply annotate observation counts of SNPs and indels
    --min-alternate-count -> Require at least 5 count of observations supporting an alternate allele within a single individual in order to evaluate the position
    --pooled-continuous -> Output all alleles which pass input filters, regardles of genotyping outcome or model
    --hwe-priors-off -> Disable estimation of the probability of the combination arising under HWE given the allele frequency as estimated by observation frequency.
    --allele-balance-priors-off ->  Disable use of aggregate probability of observation balance between alleles as a component of the priors.
    source: [Manual FreeBayes](https://vcru.wisc.edu/simonlab/bioinformatics/programs/freebayes/parameters.txt)" >> $report 
    echo -e "\nChromosomes name are going to be `grep -e ">" $viral_genome`" >> $report
    mkdir $Variant_calling ; cd $Variant_calling
    freebayes -f $viral_genome --use-mapping-quality --min-repeat-entropy 1 \
        --gvcf --haplotype-length 0 --min-alternate-count 5 --pooled-continuous \
        --hwe-priors-off --allele-balance-priors-off \
        -b $rm_duplicat_out > ${name}_Freebayes.vcf \
        2>> $logfile
    freebayes_statut=$?
    if [ $freebayes_statut -ne 0 ] ; then echo -e "\nProbleme on VC using FreeBayes: freebayes_statut=$freebayes_statut
    Stop pipeline" >> $logfile ; exit 0 ; fi
    ## Change to allow norma etc otherwise it's bug
    sed -i 's/=QR,Number=1,Type=Integer/=QR,Number=1,Type=Float/' ${name}_Freebayes.vcf
    sed -i 's/ID=QA,Number=A,Type=Integer/ID=QA,Number=A,Type=Float/' ${name}_Freebayes.vcf
    bgzip ${name}_Freebayes.vcf 2>> $logfile
    tabix -p vcf ${name}_Freebayes.vcf.gz 2>> $logfile
    check_end_chunk "freebayes" 
    check_start_chunk "bcftools" ; cd $Variant_calling
    bcftools stats ${name}_Freebayes.vcf.gz > ${name}_Freebayes.check
    echo -e "\nVCF file is normalized with bcftools norm
    More information [here](https://genome.sph.umich.edu/wiki/Variant_Normalization)"
        >> $report 
    bcftools norm -f $viral_genome \
        ${name}_Freebayes.vcf.gz -O v > ${name}_Freebayes_norma.vcf \
        2>> $logfile
    check_end_chunk "bcftools"
    check_start_chunk "vt" ; cd $Variant_calling
    echo -e "\nVCF file is decomposed with vt" >> $report 
    vt decompose_blocksub ${name}_Freebayes_norma.vcf \
        -o ${name}_Freebayes_norma_decompose.vcf \
        2>> $logfile
    check_end_chunk "vt"
    check_start_chunk "vcflib" ; cd $Variant_calling
    echo -e "\nIf multiple alleles are specified in a single record, break the record into multiple lines, preserving allele-specific INFO fields." \
        >> $report 
    vcfbreakmulti ${name}_Freebayes_norma_decompose.vcf \
        > ${name}_Freebayes_norma_decompose_multiBreak.vcf \
        2>> $logfile
    check_end_chunk "vcflib"
    check_start_chunk "bcftools" ; cd $Variant_calling
    bgzip -c ${name}_Freebayes_norma_decompose_multiBreak.vcf > $Variant_calling_out
    tabix -p vcf $Variant_calling_out
    check_end_chunk "bcftools"
fi

if [ ! -f "$annotation_VC_out" ]; then
    announcement "Annotation of the variants"
    check_start_chunk "snpeff"
    if [ ! -d $path_conda_env/snpeff/share/snpeff-4.3.1t-3/ ]; then
        echo -e "Error of the location of snpeff, this folder should exist:
        $path_conda_env/snpeff/share/snpeff-4.3.1t-3
        1) Verified that $path_conda_env correspond to the conda env
        2) Verified the location of **SnpEff** with conda install
        3) In the Script the path is hard coded, please check the path of Snpeff
        \nPipeline stop due to an issue when annotating VCF at $(date +%Y-%m-%d_%Hh%Mm%Ss)" \
            >> $logfile
        exit 0
    fi
    mkdir $annotation_VC ; cd $annotation_VC
    # Depending on the version of snpeff-4.3.1t-3, the path could change
    if [ ! -d $path_conda_env/snpeff/share/snpeff-4.3.1t-3/data/$viral_input ]; then
        mkdir -p $path_conda_env/snpeff/share/snpeff-4.3.1t-3/data/$viral_input
        echo -e "Creating a custom Database for $viral_input" >> $logfile
        # Ajout automatique dans la bdd de SnpEff : Cette partie pourra être implémenté plus tard
        ### Generation of gff
        #cat $gffFile > genes.gff
        #echo "##FASTA"  >> genes.gff
        #cat $genomefile  >> genes.gff
        cp $viral_gff genes.gff
        sed -i "s/gnl|OsHV|ORFi_1/$viral_input/" genes.gff # probleme with name of "chr" with prokka, only work because there is 1 chr
        echo -e "\nChange of 'gnl|OsHV|ORFi_1' to '$viral_input' with sed due to annotation issue\n" >> $logfile
        cp genes.gff $path_conda_env/snpeff/share/snpeff-4.3.1t-3/data/$viral_input/ #version with GFF
        #cp $viral_gb $path_conda_env/snpeff/share/snpeff-4.3.1t-3/data/viral_input/genes.gb
        cat $viral_genome  > $path_conda_env/snpeff/share/snpeff-4.3.1t-3/data/$viral_input/sequence.fa
        echo -e "\n# $viral_input $(date +%Y-%m-%d_%Hh%Mm%Ss)\n$viral_input.genome : $viral_input" \
        >> $path_conda_env/snpeff/share/snpeff-4.3.1t-3/snpEff.config
        # Construction database
        snpEff build \
            -gff3 -v $viral_input \
            &> Variant_calling.log \
            2>> $logfile
        cd $annotation_VC
    fi
    echo -e "\nAnnotated variant of $viral_input" >> $report
    snpEff -c $path_conda_env/snpeff/share/snpeff-4.3.1t-3/snpEff.config \
      -v $viral_input \
      -stats $annotation_VC/NR_genome_${name}_SnpEff_summary.html \
      $Variant_calling_out \
      > $annotation_VC_out \
      2>> $logfile
    cp $annotation_VC_out $output/output # OUTPUT
    check_end_chunk "snpeff"
    # Test de rajouté closest to make the ID of polymorphisms
fi

if [ ! -f "$VC_table_bcftools" ] || [ ! -f "$VC_table_bcftools" ] ; then
    announcement "Creation of table to assess the heterogeneity of each samples"
    check_start_chunk "snpsift"
    mkdir $VC_table ; cd $VC_table
    echo -e "\nExtratct annotation using SnpSift :\n"
    SnpSift \
        extractFields -s "," $annotation_VC_out \
        CHROM \
        POS \
        REF \
        ALT \
        "ANN[*].IMPACT:" \
        "ANN[*].EFFECT" \
        "ANN[*].HGVS_P" \
        "ANN[*].HGVS_C" \
        "ANN[*].CDNA_POS" \
        "ANN[*].CDNA_LEN" \
        "ANN[*].AA_POS" \
        "ANN[*].AA_LEN" \
        "ANN[*].DISTANCE" \
        "ANN[*].GENEID" \
        > $VC_table/${name}_ann_SnpSift_untag.csv \
            2>> $logfile
    while read p ; do
        echo -e "${p}\t${name}" >> $VC_table_SnpSift
    done < $VC_table/${name}_ann_SnpSift_untag.csv
    rm $VC_table/${name}_ann_SnpSift_untag.csv
    echo '```' >> $report
    head $VC_table_SnpSift >> $report
    echo '```' >> $report
    cp $VC_table_SnpSift $output/output # OUTPUT
    check_end_chunk "snpsift"
    # Second extraction
    check_start_chunk "bcftools" ; cd $VC_table
    echo -e "\nExtract annotation using bcftools :\n"
    bcftools query -f '%POS\t%REF\t%ALT\t%DP\t[%RO]\t[%AO]\t%TYPE\n' \
        $annotation_VC_out \
        > $VC_table/${name}_ann_Bcftools_untag.csv \
            2>> $logfile
    while read p ; do
        echo -e "${p}\t${name}" >> $VC_table_bcftools
    done < $VC_table/${name}_ann_Bcftools_untag.csv
    rm ${name}_ann_Bcftools_untag.csv
    echo '```' >> $report
    head $VC_table_bcftools >> $report
    echo '```' >> $report
    cp $VC_table_bcftools $output/output # OUTPUT
    check_end_chunk "bcftools"
fi

if [ ! -f "$output/${name}_DiVir_output.tar.gz" ]; then
    cd $output
    tar -cv output | gzip  > ${name}_DiVir_output.tar.gz   
fi

announcement "Programme Ends at : $(date +%Y-%m-%d_%Hh%Mm%Ss)"
echo -e "Divir version 0.0.1 (build 2020-05-04 ), by Jean Delmotte" >> $report