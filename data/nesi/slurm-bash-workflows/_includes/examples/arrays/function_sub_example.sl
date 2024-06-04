#!/bin/bash -e

setPaths () {

#Good practice to define paths as variables
export INDIR=/nesi/project/nesi99999/Callum/Ben/inputs/    #for your inputs
export OUTDIR=/nesi/project/nesi99999/Callum/Ben/outputs/  #For your outputs

export TEMPDIR=/nesi/nobackup/nesi99999/.tmp                   #For random working files
export FUNCPATH=/nesi/project/nesi99999/Callum/Ben/subChain.sl   #Function

export NUMBERSTEPS=6

mkdir -v -p $INDIR $OUTDIR $TEMPDIR $FUCPATH

}

##########################
#SUBMIT FUNCTION
##########################

submitStep () { 
#This is the function that submits the job
#First input is run number ($1)
#Second input is step number ($2)
#Unless called by self step number should be = 1


touch $TEMPDIR/step$1$2.sltmp
echo "Temporary file step$1$2 created"

#This misbehaves if indented
cat <<EOF > $TEMPDIR/step$1$2.sltmp
#!/bin/bash -e

#SBATCH --job-name byen_run$1_step$2 					#Name to appear in sacct 
#SBATCH --time 10:00:00 						#You can make this a product of some func if you want to dynamically set walltime. 
#SBATCH --mem-per-cpu=1500 						#1500MB is the max mem you get on the standard partition. 
#SBATCH --ntasks=1 							#MATLAB doesn't support MPI 
#SBATCH --cpus-per-task=$((($2 * 0) + 2))				#Number cores set buy function 
#SBATCH --output=${OUTDIR}byen_run$1_step$2.out 	       		#outputDir+jobname+.out for consistancy
#SBATCH --error=${OUTDIR}byen_run$1_step$2.out


source $FUNCPATH
cd $OUTDIR	#Anything relative is dumped here.

module load MATLAB
matlab -nodisplay -r "addpath('${INDIR}');training_procedure_step$2_$1" #Change directory to where MATLAB workspace outputs need to go, then call input file.

echo "Submitting step ${2} of ${numberSteps}"

#If this is the last job end, if not call the next.
if [[ $NUMBERSTEPS -gt $2 ]]
then
	echo "Submitting step $2"
	submitStep $1 $(($2 + 1))	#Submits run num and step+1
else
	echo "Final step completed."		
fi

EOF

echo "Text written to step$1$2"

sbatch $TEMPDIR/step$1$2.sltmp		#Submit this job.
echo "step$1$2 submitted to Slurm controller."

rm $TEMPDIR/step$1$2.sltmp		#Delete script (tidy)
echo "Clearing .sl file."

}

###########################
#END OF FUNCTION 
###########################
