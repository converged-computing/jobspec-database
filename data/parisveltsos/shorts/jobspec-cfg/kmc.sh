#!/bin/bash
#SBATCH --job-name=kmc                # Job name
#SBATCH --partition=sixhour               # Partition Name (Required) eeb sixhour
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=pveltsos@ku.edu   # Where to send mail 
#SBATCH --ntasks=1                          # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20gb                     # Job memory request
#SBATCH --time=0-05:59:00             # Time limit days-hrs:min:sec
#SBATCH --output=out_kmc_%j.log         # Standard output and error log


# for i in $(echo -e "IM125\nIM274\nSF\nIM62\nIM767"); do sbatch kmc.sh $i; done

echo "Running"

cd /home/p860v026/temp/$1

## PART 1 - count kmers - 16 cpu 16 memory sixhour queue - 45 min run
# this uses hard disk space instead of memory so it is best to work on the temp drive.

mkdir tmp
ls *.fasta.gz > FILES
pwd; hostname; date

kmc -k21 -t16 -m4 -ci1 -fm -cs1000000 @FILES kmc_$1 tmp =  = 
 
kmc_tools transform kmc_$1 histogram kmc_$1\_k21.hist -cx1000000

## PART 2 genomescope - 
# 
module load R/3.6
 
genomescope.R -i kmc_$1\_k21.hist -k 21 -p 2 -o genomescope_kmc_$1_\k21
 
tar -zcvf genomescope_kmc_$1_\k21.tar.gz genomescope_kmc_$1_\k21
	
## PART 3 smudgeplot - 1 cpu 300 Mb memory eeb queue 12:30 hr runtime

# module load python
# module load R/3.6
# 
# L=$(smudgeplot.py cutoff kmc_Mtilingii2_k21.hist L)
# U=$(smudgeplot.py cutoff kmc_Mtilingii2_k21.hist U)
# echo $L $U # these need to be sane values
# 
# kmc_tools transform kmc_Mtilingii2 -ci"$L" -cx"$U" dump -s kmc_Mtilingii2_L"$L"_U"$U".dump
# smudgeplot.py hetkmers -o kmc_Mtilingii2_L"$L"_U"$U" < kmc_Mtilingii2_L"$L"_U"$U".dump
# 
# smudgeplot.py plot kmc_Mtilingii2_L"$L"_U"$U"_coverages.tsv
# 
# tar -zcvf smudgeplot_kmc_Mtilingii2_k21.tar.gz smudgeplot_*
# 
# date
# 
