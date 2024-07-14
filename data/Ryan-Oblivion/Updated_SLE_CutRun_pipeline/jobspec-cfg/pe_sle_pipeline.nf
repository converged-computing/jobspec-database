// this works to get the read pairs in a new tuple form


// run this to see how it looks again
params.reads = 
"/scratch/rj931/tf_sle_project/all_sle_data/45*-Mock*cut*_{R1,R2}*.fastq.gz"

params.bg_regions = "/scratch/rj931/tf_sle_project/all_sle_data/461-IgG*cut*_{R1,R2}*.fastq.gz"

FASTP='fastp/intel/0.20.1'
FASTQC='fastqc/0.11.9'
BWA='bwa/intel/0.7.17'
SAMTOOLS='samtools/intel/1.14'
MULTIQC='multiqc/1.9'

// i want to put all the filt files into a dir so i can get them in the next process

params.outdir = 'filt_files'

params.filts = "filt_files/45*-Mock*cut*_{R1,R2}*.filt*"

//Channel
//	.watchPath('filt_files', 'create,modify')
//	.filter { it.name ==~ params.filts}
//	.until { it} 
//	.set {filt_pe}

nextflow.enable.dsl=2


PE_reads = Channel.fromFilePairs(params.reads, checkIfExists: true)

process fastp {

//tag
publishDir params.outdir, mode: 'copy'

input:
tuple val(pair_id), path(PE_reads)
// input in fastp would be ${reads[0]} and ${reads[1]}
// something i can do in fastp is to now concatenate the $pair_id'_R1.filt.fastq.gz' 
// same for output 2 in fastp use $pair_id'_R2.filt.fastq.gz'

output:
path "${pair_id}_R1.filt.fq.gz", emit: fastp_out_f
path "${pair_id}_R2.filt.fq.gz", emit: fastp_out_r

"""
#!/bin/env bash

module load $FASTP
module load $FASTQC

fastp \
-i ${PE_reads[0]} \
-I ${PE_reads[1]} \
-o $pair_id'_R1.filt.fq.gz' \
-O $pair_id'_R2.filt.fq.gz' \
--detect_adapter_for_pe \
--correction

# trying the correction option to correct mismatched base pairs in overlapped regions of paired end reads
#--trim_front1 7 \
#--trim_front2 7 \

fastqc $pair_id'_R1.filt.fq.gz' $pair_id'_R2.filt.fq.gz'

# since this script is running somewhere in the work directory I need to figure out how to find where all the qc files are

#find ./../.. -name *fastqc.zip > fastqc_files.txt

#module load $MULTIQC

#multiqc -force --file-list fastqc_files.txt --filename 'multiqc_report.html'

"""

}




params.outdir2 = "bg_filt_files"
params.bg_filts = "bg_filt_files/461-IgG*cut*_{R1,R2}*.fq.gz"

//Channel
//	.watchPath('bg_filt_files', 'create,modify')
//	.filter { it.name ==~ params.bg_filts }
//	.take(12)
//	.set {bg_reads}

bg_reads = Channel.fromFilePairs(params.bg_regions, checkIfExists: true)

process bg_fastp{

publishDir params.outdir2, mode: 'copy'

input:
tuple val(pair_id), path(bg_reads)

output:
path "${pair_id}_R1.filt.fq.gz", emit: bg_filt_out_f
path "${pair_id}_R2.filt.fq.gz", emit: bg_filt_out_r

"""
#!/bin/env bash

module load $FASTP

fastp \
-i ${bg_reads[0]} \
-I ${bg_reads[1]} \
-o $pair_id'_R1.filt.fq.gz' \
-O $pair_id'_R2.filt.fq.gz' \
--detect_adapter_for_pe \
--correction

# trying correction for the same reason explained above in the previous process
#--trim_front1 7 \
#--trim_front2 7 \
"""

}

process bwa_index {
    input:
    val ref

    output:
    path "connect_process.txt", emit: bwa_index_end

    script:
    
    """
    #!/bin/env bash
    module load $BWA

    bwa index -a $ref

    touch connect_process.txt
    """
}



filt_pe = Channel.fromFilePairs(params.filts) 
params.outdir4 = "store_normal_bam_files"

process bwa {
publishDir params.outdir4, mode: 'copy'


input:
val ref
tuple val(pair_id), path(filt_pe)
val index_end_file


output:
path "${pair_id}_sort.bam", emit: bam_file

"""
#!/bin/env bash

module load $BWA
module load $SAMTOOLS

#bwa index -a bwtsw $ref

bwa aln -t 8 $ref ${filt_pe[0]} > ${filt_pe[0]}'reads_1.sai'
bwa aln -t 8 $ref ${filt_pe[1]} > ${filt_pe[1]}'reads_2.sai'

bwa sampe $ref ${filt_pe[0]}'reads_1.sai' ${filt_pe[1]}'reads_2.sai' ${filt_pe[0]} ${filt_pe[1]} \
> ${pair_id}'aligned_reads.sam'

samtools view -b -h -q 20 ${pair_id}'aligned_reads.sam' -o $pair_id'.bam'
samtools sort $pair_id'.bam' -o $pair_id'_sort.bam' -O bam
samtools index -b $pair_id'_sort.bam'

"""
}



bg_filt = Channel.fromFilePairs(params.bg_filts)
params.outdir3 = "bg_sort_bam_files"


process bg_bwa {

publishDir params.outdir3, mode: 'copy'

input:
val ref
tuple val(pair_id), path(bg_filt)
val index_end_file

output:
path "${pair_id}_sort.bam", emit: bg_bam

"""
#!/bin/env bash

module load $BWA
module load $SAMTOOLS

#bwa index -a bwtsw $ref

bwa aln -t 8 $ref ${bg_filt[0]} > ${bg_filt[0]}'reads_1.sai'
bwa aln -t 8 $ref ${bg_filt[1]} > ${bg_filt[1]}'reads_2.sai'

bwa sampe $ref ${bg_filt[0]}'reads_1.sai' ${bg_filt[1]}'reads_2.sai' ${bg_filt[0]} ${bg_filt[1]} \
> ${pair_id}'aligned_reads.sam'

samtools view -b -h -q 20 ${pair_id}'aligned_reads.sam' -o $pair_id'.bam'

samtools sort $pair_id'.bam' -o $pair_id'_sort.bam' -O bam
samtools index -b $pair_id'_sort.bam'

"""


}



// now i want to make a process for creating tag directories
// need to make the files into a list and sort by name
// then the transpose operator will make them into a tuple
// the combine operator does it for the file paths but it is not in order

//kd_bam_list = Channel.fromPath("store_normal_bam_files/454*bam").toSortedList{it.name}
//ctr_bam_list = Channel.fromPath("store_normal_bam_files/455*bam").toSortedList{it.name}


bam_files = Channel.fromFilePairs("store_normal_bam_files/{454,455}-Mock-n{1,2,3}*_S{1,2,15,16,29,30}_L00{1,2}_sort.bam")
bam_files.view()

process bam_files {


"""
cd store_normal_bam_files

ls 454* > kd_bam.txt
ls 455* > ctr_bam.txt
paste kd_bam.txt ctr_bam.txt > kd_ctr_bam.txt 

"""

}


//bam_tuple = kd_bam_list.combine(ctr_bam_list)


//kd_bam = Channel.fromList(kd_bam_list)
//ctr_bam = Channel.fromList(ctr_bam_list)



//kd_ctr_pair = kd_bam_list.merge(ctr_bam_list)




workflow{

// look what channel.value does. it lets me use the ref multiple times
ref = Channel.value(params.ref)

//PE_reads = Channel.fromFilePairs(params.reads, checkIfExists: true)

//PE_reads.view()
//filt_pe.view()
//bg_filt.view()
//kd_bam_list.view()
//bam_tuple.view()
main:

fastp(PE_reads)
bg_fastp(bg_reads)
bwa_index(ref)
bwa(ref, filt_pe, bwa_index.out.bwa_index_end)
bg_bwa(ref, bg_filt, bwa_index.out.bwa_index_end)
//homer( bam_tuple)
//fastp.out.fastp_out_f.view()
//fastp.out.fastp_out_r.view()
//bwa.out.bam_file.view()
}
