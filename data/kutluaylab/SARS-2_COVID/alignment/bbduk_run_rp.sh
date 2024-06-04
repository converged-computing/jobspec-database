#!/bin/bash
#QSUB PARAMETERS
#PBS -l nodes=1:ppn=1,walltime=4:00:00
#PBS -m abe
#PBS -j oe
#module load fastx_toolkit


infile='/path/to/file.fastq' # the fq input file
barcode='ATATAT' # corresponding barcode sequence: typically a text file with list of barcodes (format: barcode_namd	sequence)
fd='/path/to/output/directory' # output directory
adapter='CGCGCGCG' # adapter sequence

# change directory to where bbmap is installed
cd /path/to/bbmap

tp=$fd'temp/'
logdir=$fd'logs/'
mkdir -p $tp
mkdir -p $logdir


#The below is code that reads barcodes from a barcode file into an array

declare -a indnum=()
declare -a bcnum=()
declare -a codes=()
readarray -t barcodes < $barcode

for i in "${!barcodes[@]}"
do
	line=(${barcodes[i]//;/ }) 
	indnum[i]=$i
	bcnum[i]=${line[0]}
	codes[i]=${line[1]}
done


## adapter trimming ##
## adjust the parameter values accordingly
./bbduk.sh in=$infile out=$tp'atrim.fq' literal=$adapter ktrim=r k=8 mink=7 ml=23 maxlength=52 2> $logdir'ATlog.txt'


## barcode splitting ##
for i in ${indnum[@]}
do
	./bbduk.sh in=$tp'atrim.fq' outm=$tp${bcnum[$i]}'t.fq' literal='NN'${codes[$i]} k=8 copyundefined mm=f 2> $logdir'filter'${bcnum[$i]}.txt
	./bbduk.sh in=$tp${bcnum[$i]}'t.fq' out=$fd${bcnum[$i]}.fq literal='NN'${codes[$i]} copyundefined ktrim=r k=8 mm=f ml=15 maxlength=44 2> $logdir'trim'${bcnum[$i]}.txt
done


cd $fd
rm -rf 'temp'
cd $logdir
cat * > merged-file.txt
