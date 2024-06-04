#!/bin/bash
#PBS -j oe
#PBS -m ae
#PBS -N loopTest
#PBS -M slade.allenankins@jcu.edu.au
#PBS -l select=1:ncpus=1:mem=18gb
#PBS -l walltime=00:40:00
#PBS -J 0-99

#sleep a random number of seconds (up to 60) - may alleviate too many batches running at the same time and causing I/O or downloading issues
sleep `expr $RANDOM % 20`

#set scratch directory where flac and wav files will temporarily reside
scratchdir=/scratch/jc696551/

# cd to dir where directory was created
cd $PBS_O_WORKDIR
shopt -s expand_aliases
source /etc/profile.d/modules.sh
echo "Job identifier is $PBS_JOBID"
echo "Working directory is $PBS_O_WORKDIR"

# this the path to the csv (relative to the directory on HPC storage where you launch the batch job)
export data_file=NoisyMiner/RecordingList_20230412.csv

#perform loop to analyse multiple recordings per batch job
set_size=5
start_row=$(($PBS_ARRAY_INDEX * $set_size))
end_row=$(($start_row + $set_size - 1))

#let "start_row = $((PBS_ARRAY_INDEX * $set_size))"
#let "end_row = $(($start_row + $set_size))"

echo "set size: $set_size"
echo "start row: $start_row"
echo "end row: $end_row"

for (( idx=start_row;  idx<=end_row; idx++ )); do
	echo "starting row $idx, batch $PBS_ARRAY_INDEX"
	
	export rowNum=$(($idx + 1))
	#this extracts the row of the csv that corresponds to the current PBS job within the batch
	line=$(head -n $(($idx + 2)) $data_file | tail -n 1)
	site=$(head -n $(($idx + 2)) $data_file | tail -n 1 | cut -d, -f1 --output-delimiter='|')
	recordingPath=$(head -n $(($idx + 2)) $data_file | tail -n 1 | cut -d, -f4 --output-delimiter='|')
	dirname=/scratch/jc696551/$(head -n $(($idx + 2)) $data_file | tail -n 1 | cut -d, -f6 --output-delimiter='|')
	recordingName=$(head -n $(($idx + 2)) $data_file | tail -n 1 | cut -d, -f7 --output-delimiter='|')

	#create output directory
	echo $dirname

	if [ ! -d "$dirname" ]
	then
		echo "Directory doesn't exist. Creating now"
		mkdir -p $dirname
		echo "Directory created"
	else
		echo "Directory exists"
	fi

	echo "processing row:"
	echo  $line

	echo "processing recording:"
	echo  $recordingPath
	
	#download recording to scratch directory using curl and the recordingID extracted from the csvfile
	module load curl
	cd $scratchdir/$site
	curl -u "zMIyOKUE3jdK0kS:Bioacoustics" -O -J "https://cloud.une.edu.au/public.php/webdav/$recordingPath"
	cd $PBS_O_WORKDIR
	
	#run AnalysisPrograms to create acoustic indices
	singularity run $SING/ecoacoustics-21.7.0.4.sif AnalysisPrograms audio2csv "${scratchdir}/${site}/${recordingName}" AP/ConfigFiles/Towsey.Acoustic.yml $dirname --quiet --parallel --when-exit-copy-config
	
	#run R script to generate indices
	module load R/4.1.2
	/sw/containers/R-4.1.2.sif Rscript /home/jc696551/NoisyMiner/GenerateIndices_HPC.R
	
	#remove wav file as it has now been analysed
	echo "Removing wav file"
	rm "$scratchdir/$site/$recordingName"
done