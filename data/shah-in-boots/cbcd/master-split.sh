#!/bin/sh

#SBATCH --partition=cpu-t3
#SBATCH --job-name=notes2parquet
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2		# Number of cores per task
#SBATCH --array=5
#SBATCH --error=slurm-%A-%a.err
#SBATCH --output=slurm-%A-%a.out
#SBATCH --mail-user=ashah282@uic.edu
#SBATCH --mail-type=END

# Setup and loading of modules
printf 'Load modules\n'
module load R/4.2.1-foss-2022a

# This script is complex in order to keep documentation in one place. There are
# three parts to thw script:
#
# 	1. Prepare/clean <CSV> (raw) files prior to chunking
# 	2. Split <CSV> files into more manageable chunks
# 	3. Convert <CSV> files to <PARQUET> format
#
# Currently the splitting is done by year, as the data refreshes annually
# The conversion to <PARQUET> is done by a grouping variable.
# This could be a year or an alternative approach based on the data
# Both of these tasks are placed into separate R files to be called
#
# The script below is setup in as an IF/THEN/ELSE series
# The first section can be used by SLURM by splitting tasks into batches
# The second section can be used by SLURM by splitting by year
# The third section can be used by SLURM by splitting on topic

# There are three choices: 
# 	1. "prepare" = take raw <CSV> data and prepares it for splitting
# 	2. "split" = takes raw <CSV> data and splits it by year
# 	3. "convert" = takes the split CSV data and converts to <PARQUET>

task="convert"

# 1 = PREPARE RAW
if [[ $task == "prepare" ]]
then

	# Slurm settings above will be used to help chunk the data
	# These mini-batches help with processing speed
	
	# Diagnoses
	# These need to be converted from ICD-9 to ICD-10
	# The raw file with mixed ICD codes should be called 'diagnosis-raw.csv'
	# The output file will then be 'diagnosis.csv'
	Rscript R/convert-icdCodes.R $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_COUNT
fi

# 2 = SPLIT CSV
if [[ $task == "split" ]]
then 

	# Years setup (2010 to 2023 is 14...)
	years=($(seq 2010 2023))
	year=${years[$SLURM_ARRAY_TASK_ID - 1]}

	# Past to R script with variable for years
	# Remember to update partition if using large files (e.g. procedure-records)
	# Can also select which variables are to be evaluated in Rscript
	printf "Splitting data for: $year\n"
	Rscript R/split-ccts2csv.R $year
fi

# 3 = CSV TO PARQUET
if [[ $task == "convert" ]]
then

	# Data types below = 9 overall
	types=(
		'demographics' # 1
		'diagnosis' # 2
		'labs' # 3
		'medications' # 4
		'notes' # 5
		'procedure-dates' # 6
		'procedure-reports' # 7
		'visits' # 8
		'vitals' # 9
	)

	# Type is the part of the script that will be analyzed
	type=${types[$SLURM_ARRAY_TASK_ID - 1])}

	# Rscript to run
	Rscript R/split-csv2parquet.R $type

fi

