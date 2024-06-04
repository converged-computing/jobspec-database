#PBS -S /bin/bash
#PBS -N spades_pe
#PBS -l nodes=1:ppn=8,mem=8Gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -l billto=your_project_id      # project number from which the used service units (SUs) are charged

cd $PBS_O_WORKDIR

module load SPAdes/3.5.0-goolf-1.7.20

<<'README'
    read the official SPAdes 3.5.0 Manual
    http://spades.bioinf.spbau.ru/release3.5.0/manual.html

    estimated run time on eos: ~45 minutes; max memory ~3.5Gb
        genome size 4.4Mb
        210,924 300bp read pairs

    smaller kmer values require more memory than larger values
README

################################################################################
# TODO edit these variables as needed,
threads=8       #make sure this matches your BSUB -n value
max_memory=8    #max memory used in Gb, make sure this matches your BSUB mem value
kmers=91,95,101  #SPAdes will select the best build of the kmers you provide; must be odd number; max kmer=127
output_dir='build_spades_1pe'
yaml_file='conf_spades_1pe.yaml'

# Using yaml format allows you to indicate orientation, and type of reads:
#   orientation: fr (forward reverse), rf (reverse forward), ff (forward forward)
#   type:   paired-end, mate-pairs, hq-mate-pairs, single
#           pacbio, nanopore, sanger, trusted-contigs, untrusted-contigs
# TODO Edit the names of your files in the appropriate lines below
echo "
[
  {
    orientation: 'fr',
    type: 'paired-end',
    left reads: [
      '../../../data/sra/m_tuberculosis/ERR551611_pe_1.fastq.gz'
    ],
    right reads: [
      '../../../data/sra/m_tuberculosis/ERR551611_pe_2.fastq.gz'
    ]
  }
]
" > $yaml_file
#
################################################################################
# command to run with defaults and the --cafeful option
spades.py --threads $threads --careful --dataset $yaml_file --memory $max_memory -k $kmers -o $output_dir

# example of how to restart run with mismatch correctoin to try to reduce the number of mismatches and short indels
#spades.py --threads $threads --careful --restart-from mc --memory $max_memory -k $kmers -o $output_dir

qstat -f $PBS_JOBID
