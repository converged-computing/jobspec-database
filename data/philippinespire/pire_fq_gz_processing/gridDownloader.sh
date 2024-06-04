#!/bin/bash

#SBATCH --job-name=gridDown
#SBATCH -o gridDown-%j.out
#SBATCH --time=00:00:00
#SBATCH --exclusive
#SBATCH --ntasks=40
#SBATCH -p main

# this script download files (meant for fq.gz files) from TAMUCC "whebshare/grid"
# contact: Eric e1garcia@odu.edu

# EXECUTE
# First, locate the grid parent directory (which is just a website link) where the files will be downloaded from. Usually Sharon will post this in the species slack channel.
# visually check in your browser that there is a file named tamucc_files.txt, which list the rest of the files in that directory
# There are 3 protocols. Try in this order, wget, wget1, wget2, curl, till you get not errors
# (size errors are identified in this script and reported at the end of the log. Format errors are identified by checkFQ.sh)  
# execute as "sbatch grid_wget.sh <outdir> <link-to-files> <wget|wget1|wget2|curl>

#module load parallel
export SINGULARITY_BIND=/home/e1garcia  #odu

# User defined variables
OUTDIR=$1
LINK=$(echo $2 | sed 's/\/$//')
PROTOCOL=$3

cd $OUTDIR

if [[ "$PROTOCOL" == "wget" ]]; then
        COMMAND="wget -c -P"
	echo -e "Downloading using:\nprotocol= $PROTOCOL\ncommand= $COMMAND\n\n"
elif [[ "$PROTOCOL" == "wget1" ]]; then
        COMMAND="wget --secure-protocol=TLSv1_2 -c -P"    #yes, TLSv1_2 first
        echo -e "Downloading using:\nprotocol= $PROTOCOL\ncommand= $COMMAND\n\n"
elif [[ "$PROTOCOL" == "wget2" ]]; then
        COMMAND="wget --secure-protocol=TLSv1_1 -c -P"
	echo -e "Downloading using:\nprotocol= $PROTOCOL\ncommand= $COMMAND\n\n"
elif [[ "$PROTOCOL" == "curl" ]]; then
        COMMAND="curl -L -O -C - --output-dir"
	echo -e "Downloading using:\nprotocol= $PROTOCOL\ncommand= $COMMAND\n\n"
elif [[ -z "$PROTOCOL" ]]; then
        COMMAND="wget -c -P"
	echo -e "Protocol not specified. Downloading using:\nprotocol=wget\ncommand=$COMMAND\n\nIf you meant to run a different protocol please provite it as the third argument\nProtocol options= wget, wget1, wget2, curl. Try in this order\n\n."
else
        echo -e "Protocol not recognized. Options are: wget, wget1, wget2, curl. Please provide one of these as the third argument. Try them in the same order\n\n"
fi 


# downlownd the list of lifes
$COMMAND $OUTDIR $LINK/tamucc_files.txt
NCOL=$(cat tamucc_files.txt | tail -n1 | awk '{print NF}')

# Check if the tamucc_files was created with ls or ls -lthr, and use that list to download files
if [[ $NCOL -eq 9 ]]
then 
# use that list to download all files in parallel.
cat $OUTDIR/tamucc_files.txt | grep '^[-d]' | tr -s " " | cut -d " " -f9 | parallel --no-notice -kj20 $COMMAND $OUTDIR $LINK/{}
echo -e "\nFirst download completed.\n\nDownloading again in case partial downloads\n" 

# Second download to check for partial downloads
cat $OUTDIR/tamucc_files.txt | grep '^[-d]' | tr -s " " | cut -d " " -f9 | parallel --no-notice -kj20 $COMMAND $OUTDIR $LINK/{}
echo -e "\nSecond download completed" 

# Checking sizes of files from source and lpwd
#echo -e "\nComparing file sizes from source and lpwd with (grep -vf)"

cd $OUTDIR

#Update August 2023, Size comparison is not done by checkFQ.sh
#cat tamucc_files.txt | grep '[zv]$' | tr -s " " | cut -d " " -f5,9 > tamucc_gzfiles
#ls -lh *[zv] | tr -s " " | cut -d " " -f5,9 > odu_gzfiles

#grep -vf tamucc_gzfiles odu_gzfiles > files_wDiff_sizes

#if [[ -s files_wDiff_sizes ]]
#then 
#        echo -e "\nFiles with different sizes detected. Offending file(s) printed in files_wDiff_sizes. Please check files_wDiff_sizes and compare tamucc_files.txt with current downloaded data"
#	rm tamucc_gzfiles odu_gzfiles
#else 
#	echo -e "\nNo size mismatch in files was detected"
#        rm tamucc_gzfiles odu_gzfiles files_wDiff_sizes
#fi

echo -e "\nIf your download did not work at all, click on the link to the files and visually check in your browser that there is a file named tamucc_files.txt or contact Eric e1garcia@odu.edu"

elif [[ $NCOL -eq 1 ]] 
then
# use that list to download all files in parallel.
cat $OUTDIR/tamucc_files.txt | parallel --no-notice -kj20 $COMMAND $OUTDIR $LINK/{}
echo -e "\nFirst download completed.\n\nDownloading again in case partial downloads\n" 

# Second download to check for partial downloads
cat $OUTDIR/tamucc_files.txt | parallel --no-notice -kj20 $COMMAND $OUTDIR $LINK/{}
echo -e "\nSecond download completed" 

# Checking sizes of files from source and lpwd
echo -e "\nThe tamucc_files.txt does not have file size information (it was created with a simple ls) so this script cannot compare the size of files after download.\nPlease visually compare the size of downloaded files with what is posted in the web browser from the http link\n\nIf you have a lot of files, it might be worth asking Sharon or someone at TAMUCC to recreate the tamucc_files.txt with an ls -ltrh, in which case this script will automatically check the size of files before and after download"

echo -e "\nIf your download did not work at all, click on the link to the files and visually check in your browser that there is a file named tamucc_files.txt"
fi

echo  -e "\ngridDownloader.sh is done. Please run checkFQ.sh to check the size and format of files"
