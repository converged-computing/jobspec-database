#!/bin/bash

echo -e "\033[1;34m    ___   __   _   _   _____   _    _   _____        _____   ____   ____ "
echo -e "\033[1;34m   |  _| |  \ | | | | |  ___| | \  / | |  _  |  __  |  _  | |  __| |  _ \ "
echo -e "\033[1;34m   | \   |   \| | | | | | __  |  \/  | | |_| | |__| | | | | | |    | | \ \ "
echo -e "\033[1;34m   | /_  | |\   | | | | |_\ \ | |\/| | | | | |      | |_| | | |__  | |_/ / "
echo -e "\033[1;34m   |___| |_| \__| |_| |_____| |_|  |_| |_| |_|      |_____| |____| |____/ "
      
echo ""
echo -e "\033[1;34m                 *** Imaging Transcriptomics Project ***"
echo ""
echo -e "\033[0m    $USER, welcome and many thanks for contributing to this project!!"
echo ""
echo ""


# Define the name of the Singularity image
image_name="morphometry.sif"

# Ask for the path to build the image
echo -e "\033[1;34m[prompt]\033[0m Enter the directory path to build the image or press [Enter] for the current directory: \c" 
read image_path
        
# If the image path is empty, set it to the current directory
if [ -z "$image_path" ]; then
    image_path="."
fi

# Check if the image file exists in the current directory
if [ ! -f "$image_path/$image_name" ]; then
    echo -e "\033[1;34m[note]\033[0m Image $image_name does not exist in the selected directory - we will write and submit a sbatch script to download it"

    script_name_build="build_image.sh"

    # Check if the build_image.sh file already exists
    if [ -f "$script_name_build" ]; then
        echo -e "\033[1;34m[note]\033[0m $script_name_build script already exists."
        echo -e "\033[1;34m[prompt]\033[0m Do you want to use it [y] or generate a new one [n]? \c" 
        read use_existing
    fi

    if [ ! -f "$script_name_build" ] || [ "$use_existing" = "N" ] || [ "$use_existing" = "n" ]; then
        # Delete the old script file
        if [ -f $script_name_build ]; then rm -f $script_name_build; fi

        # Collect sbatch flags iteratively
        echo -e "\033[1;34m[prompt]\033[0m Enter the #SBATCH flags for building the $script_name_build script. Press 'q' when finished."
		sbatch_flags_build=()
		while true; do
    		read flag
    		if [ "$flag" = "q" ]; then
        		break
    		elif [[ "$flag" != --* ]]; then
        		echo -e "\033[31mInvalid input. Flag should start with '--'.\033[0m" # Prints the message in red
        		continue
    		else
        		sbatch_flags_build+=("$flag")
    		fi
		done

        # Generate a sbatch script
        echo "#!/bin/bash" > $script_name_build
        for flag in "${sbatch_flags_build[@]}"; do
            echo "#SBATCH $flag" >> $script_name_build
        done
        #echo "module load singularity" >> $script_name_build
        echo "singularity build "$image_path/$image_name" docker://csleo/morphometry:latest" >> $script_name_build

    fi

    # Execute the sbatch script
    chmod +x $script_name_build
    jobid=$(sbatch $script_name_build | awk '{print $4}')

    # Define array for loading animation
    loading_anim=("." ".." "...")

    # Counter for animation state
    counter=0

    # Check job status
    squeue -j $jobid | grep -q $jobid
    job_status=$?

    # Wait for the job to complete
    while [ $job_status -eq 0 ]; do
        echo -ne "Job $jobid is not yet complete. Waiting${loading_anim[$counter]}\r"
        sleep 1
        # Reset counter if it reaches the length of loading_anim
        if [ $counter -ge ${#loading_anim[@]} ]; then
            counter=0
        fi
        let counter++
        # Recheck job status
        squeue -j $jobid | grep -q $jobid
        job_status=$?
    done

    echo -e "\033[1;34m[note]\033[0m Job $jobid has completed.\n"

else
    echo -e "\033[1;34m[note]\033[0m Image $image_name exists in the selected directory."
fi

# Check if the image file exists in the selected directory
if [ ! -f "$image_path/$image_name" ]; then
    echo -e "\033[1;34m[note]\033[0m The image $image_name does not exist in the selected directory."
    echo -e "\033[1;34m[note]\033[0m Please inspect the slurm-$jobid.out file for any errors."
    exit 1
fi

# Check if only run_image.sh is in the pwd
if [ -f "run_image.sh" ] && [ ! -f "run_image_sub.sh" ] && [ ! -f "run_image_stats.sh" ]; then
    echo -e "\033[1;34m[note]\033[0m A script to run the pipeline as a single job already exists."
    echo -e "\033[1;34m[prompt]\033[0m Do you want to use it [y] or generate a new one [n]? \c"
    read use_existing_run_image

	if [[ "$use_existing_run_image" == "n" || "$use_existing_run_image" == "N" ]]; then
    	# Delete the old script file
    	rm run_image.sh
    fi
elif [ ! -f "run_image.sh" ] && [ -f "run_image_sub.sh" ] && [ -f "run_image_stats.sh" ]; then
    # Check if only run_image_sub.sh and run_image_stats.sh are in the pwd
    echo -e "\033[1;34m[note]\033[0m The scripts to run the pipeline as an array job already exist."
    echo -e "\033[1;34m[prompt]\033[0m Do you want to use them [y] or generate new ones [n]? \c"
    read use_existing_sub_stats

	if [[ "$use_existing_sub_stats" == "n" || "$use_existing_sub_stats" == "N" ]]; then
    	# Delete the old script files
    	rm run_image_sub.sh run_image_stats.sh
    fi
elif [ ! -f "run_image.sh" ] && [ ! -f "run_image_sub.sh" ] && [ ! -f "run_image_stats.sh" ]; then
	echo -e "\033[1;34m[note]\033[0m No script(s) to run the pipeline found in the current directory."
else
    echo -e "\033[1;34m[note]\033[0m Found a combination of scripts to run the pipeline that doesn't match expected scenarios."
    echo -e "\033[1;34m[note]\033[0m All scripts to run the pipeline found in the current directory will be deleted and new ones will be generated."
    rm run_image*.sh 2> /dev/null
fi

if [ -z "$(ls | grep 'run_image.*\.sh')" ]; then
    #echo "No run_image*.sh scripts found in the current directory. New sbatch script(s) for running the pipeline will be generated."

	# Ask the user if they want to run the job as an array
	echo -e "\033[1;34m[prompt]\033[0m Do you want to run the job as an array [y/n]? \c"
    read run_as_array
	if [[ "$run_as_array" == "y" || "$run_as_array" == "Y" ]]; then
		array_job=true
		echo -e "\033[1;34m[note]\033[0m To run the pipeline as an array job, two sbatch scripts will be generated: one for running the preprocessing workflow and one for running the morphometric stats workflow. The handler script will take care of generating and running the two instances."
	else
		array_job=false
	fi

	# Collect sbatch flags iteratively
	if [ "$array_job" = true ]; then
		echo -e "\033[1;34m[prompt]\033[0m Enter the #SBATCH flags for building the preprocessing workflow script. You should enter the array flag here. Press 'q' when finished."
		sbatch_flags_preproc=()
		while true; do
		    read flag
		    if [ "$flag" = "q" ]; then
		        break
		    elif [[ "$flag" != --* ]]; then
		        echo -e "\033[31mInvalid input. Flag should start with '--'.\033[0m" # Prints the message in red
		        continue
		    else
		        sbatch_flags_preproc+=("$flag")
		    fi
		done
		echo -e "\033[1;34m[prompt]\033[0m Enter the #SBATCH flags for building the morphometric stats workflow script. Press 'q' when finished."
		sbatch_flags_stats=()
		while true; do
		    read flag
		    if [ "$flag" = "q" ]; then
		        break
		    elif [[ "$flag" != --* ]]; then
		        echo -e "\033[31mInvalid input. Flag should start with '--'.\033[0m" # Prints the message in red
		        continue
		    else
		        sbatch_flags_stats+=("$flag")
		    fi
		done
	else
		echo -e "\033[1;34m[prompt]\033[0m Enter the #SBATCH flags for building the script. Press 'q' when finished."
		sbatch_flags_run=()
		while true; do
		    read flag
		    if [ "$flag" = "q" ]; then
		        break
		    elif [[ "$flag" != --* ]]; then
		        echo -e "\033[31mInvalid input. Flag should start with '--'.\033[0m" # Prints the message in red
		        continue
		    else
		        sbatch_flags_run+=("$flag")
		    fi
		done
	fi


    while true; do
    	# Ask for the path to the NIfTi files directory
    	echo -e "\033[1;34m[prompt]\033[0m Enter the absolute path to the directory containing the NIfTi files: \c"
        read nifti_dir
    
    	# Check if the directory exists and the path is absolute
    	if [[ "$nifti_dir" == /* ]] && [ -d "$nifti_dir" ]; then
        	break
    	else
        	echo -e "\033[31mInvalid absolute path for the directory containing the NIfTi files. Please enter an absolute path and try again.\033[0m"  
    	fi
	done

    while true; do
    	# Ask for the structure of the NIfTI files directory
    	echo -e "\033[1;34m[prompt]\033[0m Enter the structure of the directory containing the NIfTI files [raw or bids]: \c"
        read nifti_dir_struc

    	# Check if the input is 'raw' or 'bids'
    	if [[ "$nifti_dir_struc" == "raw" ]] || [[ "$nifti_dir_struc" == "bids" ]]; then
        	break
    	else
        	echo -e "\033[31mInvalid input. Please enter 'raw' or 'bids'.\033[0m" 
    	fi
	done

	while true; do
        # Ask for the directory of previous recon-all runs, if any
        echo -e "\033[1;34m[prompt]\033[0m Enter the absolute path to the directory of previous recon-all runs [press enter if there isn't]: \c"
        read recon_dir

        # Check if the directory exists
        if [[ -z "$recon_dir" ]] || ( [ -d "$recon_dir" ] && [[ "$recon_dir" == /* ]] ); then
            break
        else
            echo -e "\033[31mInvalid absolute path for the directory of previous recon-all runs. Please try again.\033[0m"  
        fi
    done


    while true; do
    	# Ask for the number of processors to use
    	echo -e "\033[1;34m[prompt]\033[0m Enter the number of threads to use for the analysis: \c"
    	read n_procs
    
    	# Check if the input is a number
    	if [[ "$n_procs" =~ ^[0-9]+$ ]]; then
        	break
    	else
        	echo -e "\033[31mInvalid input. Please enter a number.\033[0m" 
    	fi
	done
	
	# Create enigma-ocd directory to store output
	if [ ! -d enigma_ocd ]; then mkdir -p enigma_ocd; fi
	
	if [ "$array_job" = true ]; then

		# Generate a sbatch script to run at the subject_level
    	echo "#!/bin/bash" > run_image_sub.sh
    	echo "#!/bin/bash" > run_image_stats.sh
    	for flag in "${sbatch_flags_preproc[@]}"; do
        	echo "#SBATCH $flag" >> run_image_sub.sh
    	done
		for flag in "${sbatch_flags_stats[@]}"; do
        	echo "#SBATCH $flag" >> run_image_stats.sh
    	done
		if [ "$nifti_dir_struc" == "bids" ]; then 
			echo "participants=($(find "$nifti_dir" -name "sub-*" -type d -printf "%f\n"))" >> run_image_sub.sh
		elif [ "$nifti_dir_struc" == "raw" ]; then 
    		echo "participants=($(find "$nifti_dir" -type f \( -name "*.nii" -o -name "*.nii.gz" \)))" >> run_image_sub.sh
		fi
		echo "PARTICIPANT_LABEL=\${participants[\$((SLURM_ARRAY_TASK_ID - 1))]}" >> run_image_sub.sh
		if [ -z "$recon_dir" ]; then
            echo "singularity run --cleanenv --bind $nifti_dir:/nifti_dir,$(pwd)/enigma_ocd:/enigma_ocd $image_path/$image_name \
          /nifti_dir /enigma_ocd --data-dir-structure $nifti_dir_struc --participant-label \$PARTICIPANT_LABEL --work-dir /enigma_ocd --n-procs $n_procs --preproc-only" >> run_image_sub.sh
    	else
        	echo "singularity run --cleanenv --bind $nifti_dir:/nifti_dir,$(pwd)/enigma_ocd:/enigma_ocd,$recon_dir:/recon_dir $image_path/$image_name \
      /nifti_dir /enigma_ocd --data-dir-structure $nifti_dir_struc --participant-label \$PARTICIPANT_LABEL --work-dir /enigma_ocd --recon-all-dir /recon_dir --n-procs $n_procs --preproc-only" >> run_image_sub.sh

    	fi
		if [ -z "$recon_dir" ]; then
        	echo "singularity run --cleanenv --bind $nifti_dir:/nifti_dir,$(pwd)/enigma_ocd:/enigma_ocd $image_path/$image_name \
              /nifti_dir /enigma_ocd --data-dir-structure $nifti_dir_struc --work-dir /enigma_ocd --n-procs $n_procs --group-stats-only" >> run_image_stats.sh
    	else
        	echo "singularity run --cleanenv --bind $nifti_dir:/nifti_dir,$(pwd)/enigma_ocd:/enigma_ocd,$recon_dir:/recon_dir $image_path/$image_name \
              /nifti_dir /enigma_ocd --data-dir-structure $nifti_dir_struc --work-dir /enigma_ocd --recon-all-dir /recon_dir --n-procs $n_procs --group-stats-only" >> run_image_stats.sh
    	fi

	else

		# Generate a sbatch script
    	echo "#!/bin/bash" > run_image.sh
    	for flag in "${sbatch_flags_run[@]}"; do
        	echo "#SBATCH $flag" >> run_image.sh
    	done
    	if [ -z "$recon_dir" ]; then
        	echo "singularity run --cleanenv --bind $nifti_dir:/nifti_dir,$(pwd)/enigma_ocd:/enigma_ocd $image_path/$image_name \
              /nifti_dir /enigma_ocd --data-dir-structure $nifti_dir_struc --work-dir /enigma_ocd --n-procs $n_procs" >> run_image.sh
    	else
        	echo "singularity run --cleanenv --bind $nifti_dir:/nifti_dir,$(pwd)/enigma_ocd:/enigma_ocd,$recon_dir:/recon_dir $image_path/$image_name \
              /nifti_dir /enigma_ocd --data-dir-structure $nifti_dir_struc --work-dir /enigma_ocd --recon-all-dir /recon_dir --n-procs $n_procs" >> run_image.sh
    	fi

	
	fi
	
fi

# Get $array_job in case it was not set before
if [ -f run_image_sub.sh ] && [ -f run_image_stats.sh ]; then
    array_job=true
fi

# Make the script executable
if [ "$array_job" = true ]; then
    chmod +x run_image_sub.sh
    chmod +x run_image_stats.sh
else
    chmod +x run_image.sh
fi

# Execute the sbatch script
if [ "$array_job" = true ]; then
    job_id=$(sbatch --parsable run_image_sub.sh)
    
    # Check if the job was submitted successfully
    if [ -z "$job_id" ]; then
        echo -e "\033[31mThere was a problem submitting the script. Please check your configurations and try again.\033[0m"
        exit 1
    fi

    echo -e "\033[1;34m[note]\033[0m Array job submitted with ID $job_id. Waiting for it to finish..."

    # Wait for the array job to finish
    while [ -n "$(squeue -j $job_id -h)" ]; do 
        sleep 10
    done

    # Check if the array job ran successfully
    sacct_output=$(sacct -j $job_id --format=state%7 -n)
    sacct_exit_status=$?

    if [ $sacct_exit_status -ne 0 ]; then
        echo -e "\033[31mThere was a problem retrieving the status of the array job. Please check your configurations and try again.\033[0m"
        exit 1
    elif [[ "$sacct_output" == *"COMP"* ]]; then
        echo -e "\033[1;34m[note]\033[0m Array job completed successfully. Submitting the stats job..."
        stats_job_id=$(sbatch --parsable run_image_stats.sh)
        stats_job_exit_status=$?
    
        if [ $stats_job_exit_status -ne 0 ]; then
            echo -e "\033[31mThere was a problem submitting the stats job. Please check your configurations and try again.\033[0m"
            exit 1
        fi
    else
        echo -e "\033[31mArray job did not complete successfully. Please check the slurm-$job_id.out file for more information.\033[0m"
        exit 1
    fi
    
else

    # Submit the image job and save its job ID and exit status
    job_id=$(sbatch --parsable run_image.sh)
    job_exit_status=$?
    
    # Check if the image job was submitted correctly
    if [ $job_exit_status -ne 0 ]; then
        echo -e "\033[31mThere was a problem submitting the image job. Please check your configurations and try again.\033[0m"
        exit 1
    else
        echo -e "\033[1;34m[note]\033[0m Job submitted with ID $job_id. Waiting for it to finish..."
    
        # Wait for the job to finish
        while [ -n "$(squeue -j $job_id -h)" ]; do 
            sleep 10
        done
    
        # Check if the job ran successfully
        sacct_output=$(sacct -j $job_id --format=state%7 -n)
        sacct_exit_status=$?
        
        if [ $sacct_exit_status -ne 0 ]; then
            echo -e "\033[31mThere was a problem retrieving the status of the job. Please check your configurations and try again.\033[0m"
            exit 1
        elif [[ "$sacct_output" == *"COMP"* ]]; then
            echo -e "\033[1;34m[note]\033[0m Job completed successfully."
        else
            echo -e "\033[31mJob did not complete successfully. Please check the slurm-$job_id.out file for more information.\033[0m"
            exit 1
        fi
    fi
fi
