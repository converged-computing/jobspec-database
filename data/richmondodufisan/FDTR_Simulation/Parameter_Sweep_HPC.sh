#!/bin/bash

module purge
module use /software/spack_v20d1/spack/share/spack/modules/linux-rhel7-x86_64/
module load singularity
module load mpi/mpich-4.0.2-gcc-10.4.0

# Function to check if there are any jobs in the Slurm queue
function check_squeue() {
    squeue_output=$(squeue -t PD,R -u vtw1026 -h)  # Replace with your actual username
    if [ -z "$squeue_output" ]; then
        return 0  # No jobs in the queue
    else
        return 1  # Jobs are in the queue
    fi
}


# Set the maximum number of times to submit the batch job
#n_iterations=60
n_iterations=60

# Set the number of periods each job/sweep should solve for
n_periods_per_job=0.5

# Initial timestep
start_val=0.0

# Submit the initial parameter sweep

#Original file name
og_filename="FDTR_input"
extension=".i"

og_mesh_script="FDTR_mesh"
og_mesh_ext=".py"

first_period=$n_periods_per_job

# Define the range of values you want to loop over

#x0_vals_num=("0")

#freq_vals_num=("1e6" "2e6")

#theta_vals_num=("0")

x0_vals_num=("-15" "-10" "-9" "-8" "-7" "-6" "-5" "-4" "-3" "-2" "-1" "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "15")

freq_vals_num=("1e6" "2e6" "4e6" "6e6" "10e6")

theta_vals_num=("0" "75")


# Loop over values
for x0_val_num in "${x0_vals_num[@]}"; do
	for theta_val_num in "${theta_vals_num[@]}"; do
	
		# Replace the x0_val value in the mesh script
		sed -i "s/\(xcen\s*=\s*\)[0-9.eE+-]\+/\1$x0_val_num/g" "${og_mesh_script}${og_mesh_ext}"
		
		# Replace the theta_val value in the mesh script
		sed -i "0,/theta\s*=\s*[0-9.eE+-]\+/{s//theta = $theta_val_num/}" "${og_mesh_script}${og_mesh_ext}"
		
		# Replace the mesh name in the mesh script
		new_mesh_name="${og_mesh_script}_x0_${x0_val_num}_theta_${theta_val_num}.msh"
		sed -i "0,/newMeshName = [^ ]*/s/newMeshName = [^ ]*/newMeshName = \"$new_mesh_name\"/" "${og_mesh_script}${og_mesh_ext}"	
		
		#echo "$new_mesh_name"
		
		# Make new 3D mesh
		#python3 FDTR_mesh.py >> gmsh_output.txt &
		#wait
		
		# Submit Job
		#sbatch --wait FDTR_Batch_gmsh.sh
		
		echo "Mesh Generated, x0 = ${x0_val_num}, theta = ${theta_val_num}"
		
		for freq_val_num in "${freq_vals_num[@]}"; do
			# Create a new filename by appending x0_val to the original filename
			new_filename="${og_filename}_theta_${theta_val_num}_freq_${freq_val_num}_x0_${x0_val_num}_v1.i"

			# Copy the original input file to the new filename
			cp "$og_filename$extension" "$new_filename"
			
			# Replace the theta_val value in the new input file
			sed -i "s/\(theta_deg\s*=\s*\)[0-9.eE+-]\+/\1$theta_val_num/g" "$new_filename"
			
			# Replace the freq_val value in the new input file
			sed -i "s/\(freq_val\s*=\s*\)[0-9.eE+-]\+/\1$freq_val_num/g" "$new_filename"

			# Replace the x0_val value in the new input file
			sed -i "s/\(x0_val\s*=\s*\)[0-9.eE+-]\+/\1$x0_val_num/g" "$new_filename"
			
			# Replace the mesh in the MOOSE script
			sed -i "0,/file = [^ ]*/s/file = [^ ]*/file = \"$new_mesh_name\"/" "$new_filename"
			
			# Replace the end period
			sed -i "s/\(end_period\s*=\s*\)[0-9.eE+-]\+/\1$first_period/g" "$new_filename"
			
			# Replace the input file in the Batch script
			sed -i "0,/script_name=[^ ]*/s/script_name=[^ ]*/script_name=\"$new_filename\"/" "FDTR_Batch_MOOSE.sh"
			
			freq_noexp=$(python3 -c "import math; print(int($freq_val_num*1e-6))")
			
			# Replace the job name
			sed -E -i "s/(#SBATCH --job-name=)[^[:space:]]+/\1${x0_val_num}${freq_noexp}${theta_val_num}/" "FDTR_Batch_MOOSE.sh"

			# Submit job
			#sbatch FDTR_Batch_MOOSE.sh
		done
	done
done

#submission_count=1

# Change this value to continue from nth simulation
submission_count=14

#o_start=$start_val

# Change this value to the START period of the nth simulation (nth_simulation * period_per_sim) - period_per_sim
o_start=6.5

#Initial input file name
init_filename="FDTR_input"

#Original file name
og_filename="FDTR_input_restart"
extension=".i"

# RESTART Loop
while [ $submission_count -lt $n_iterations ]; do
    if check_squeue; then
		echo "No jobs in the queue. Submitting batch job script..."

		mv *.e /scratch/vtw1026/		

		# Delete mesh files from older submission
                if [ $submission_count -gt 1 ]; then
                        older_submission=$((submission_count - 1))
                	rm /scratch/vtw1026/*v${older_submission}*e
                fi
		

		o_ver=$(echo "$submission_count" | bc -l)
		n_ver=$(echo "$submission_count + 1" | bc -l)
		former_sim_ver="v${o_ver}"
		new_sim_ver="v${n_ver}"
		
		rm *${former_sim_ver}*i

		o_stop=$(echo "$o_start + $n_periods_per_job" | bc -l)
		n_stop=$(echo "$o_stop + $n_periods_per_job" | bc -l)
		
		old_start=$o_start
		old_end=$o_stop
		new_start=$o_stop
		new_end=$n_stop



		# Loop over values
		for x0_val_num in "${x0_vals_num[@]}"; do
			for theta_val_num in "${theta_vals_num[@]}"; do
				for freq_val_num in "${freq_vals_num[@]}"; do

					# Create a new filename by appending x0_val to the original filename
					new_filename="${init_filename}_theta_${theta_val_num}_freq_${freq_val_num}_x0_${x0_val_num}_${new_sim_ver}.i"

					# Copy the original input file to the new filename
					cp "$og_filename$extension" "$new_filename"
					
					# Replace the theta_val value in the new input file
					sed -i "s/\(theta_deg\s*=\s*\)[0-9.eE+-]\+/\1$theta_val_num/g" "$new_filename"
					
					# Replace the freq_val value in the new input file
					sed -i "s/\(freq_val\s*=\s*\)[0-9.eE+-]\+/\1$freq_val_num/g" "$new_filename"

					# Replace the x0_val value in the new input file
					sed -i "s/\(x0_val\s*=\s*\)[0-9.eE+-]\+/\1$x0_val_num/g" "$new_filename"
					
					# Replace the mesh in the MOOSE script
					scratch_path="\/scratch\/vtw1026\/"
					former_sim_output="${init_filename}_theta_${theta_val_num}_freq_${freq_val_num}_x0_${x0_val_num}_${former_sim_ver}_out.e"
					sed -i "0,/file = [^ ]*/s/file = [^ ]*/file = \"$scratch_path$former_sim_output\"/" "$new_filename"
					
					
					############# Replacing end and start periods #############
					
					sed -i "s/\(prev_start\s*=\s*\)[0-9.eE+-]\+/\1$old_start/g" "$new_filename"
					sed -i "s/\(prev_end\s*=\s*\)[0-9.eE+-]\+/\1$old_end/g" "$new_filename"
					sed -i "s/\(start_period\s*=\s*\)[0-9.eE+-]\+/\1$new_start/g" "$new_filename"
					sed -i "s/\(end_period\s*=\s*\)[0-9.eE+-]\+/\1$new_end/g" "$new_filename"
					
					############# END Replacing end and start periods #############
					
					# Replace the input file in the Batch script
					sed -i "0,/script_name=[^ ]*/s/script_name=[^ ]*/script_name=\"$new_filename\"/" "FDTR_Batch_MOOSE.sh"
					
					freq_noexp=$(python3 -c "import math; print(int($freq_val_num*1e-6))")
					
					# Replace the job name
					sed -E -i "s/(#SBATCH --job-name=)[^[:space:]]+/\1${x0_val_num}${freq_noexp}${theta_val_num}/" "FDTR_Batch_MOOSE.sh"

					# Submit job
					sbatch FDTR_Batch_MOOSE.sh			
				done
			done
		done	
		
		submission_count=$((submission_count + 1))
		o_start=$o_stop
    else
        echo "Jobs are in the queue. Currently solving iteration no ${submission_count}. Waiting..."
    fi

    # Sleep for a while before checking again (e.g., 5 minutes)
    sleep 300
done
