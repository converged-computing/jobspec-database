#!/bin/bash

# Job script template
read -r -d '' JOB_SCRIPT_TEMPLATE << 'EOF'
#!/bin/bash
#PBS -N MonteCarlo_JOB
#PBS -l nodes=NODES:ppn=PPN:p100
#PBS -l walltime=00:10:00
#PBS -j oe

# ensure we can access the file
cd $PBS_O_WORKDIR

# move into orbslam for the final command
cd orb_slam3

# finally do the actual orbslam with sample video
./Examples/Stereo/stereo_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml ~/Datasets/EuRoc/MH01 ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt dataset-MH01_stereo

# save results into results folder
echo 'saving results now'
cd $PBS_O_WORKDIR && mkdir -p MonteCarlo_JOB
mv orb_slam3/LocalMapTimeStats.txt orb_slam3/ExecMean.txt orb_slam3/f_dataset-MH01_stereo.txt orb_slam3/SessionInfo.txt orb_slam3/kf_dataset-MH01_stereo.txt orb_slam3/LBA_Stats.txt orb_slam3/TrackingTimeStats.txt MonteCarlo_JOB/
echo 'saved in MonteCarlo_JOB!'
EOF

# Array of processor counts
processors=(1 2 4 8 16)

# Maximum processors per node
max_ppn=16

# Initialize Prev Job
prev_job_id=""

# Loop over each processor count
for p in "${processors[@]}"
do
    # Determine how many nodes and ppn are needed
    if [ $p -le $max_ppn ]; then
        nodes=1
        ppn=$p
    else
        nodes=$(($p / $max_ppn))
        ppn=$max_ppn
        # Handle cases where processor count isn't exactly divisible by max_ppn
        if [ $(($p % $max_ppn)) -ne 0 ]; then
            nodes=$(($nodes + 1))
        fi
    fi

    nums="cuda"

    # Generate job script for this combination
    job_script="job_${p}_procs_${nums}.pbs"
    echo "${JOB_SCRIPT_TEMPLATE}" > $job_script
    sed -i "s/NODES/$nodes/g" $job_script
    sed -i "s/PPN/$ppn/g" $job_script
    sed -i "s/PROCS/$p/g" $job_script
    sed -i "s/NUMS/$nums/g" $job_script
    sed -i "s/MonteCarlo_JOB/Tuckoo_${p}procs_${nums}/g" $job_script

    # Submit job script, saving it's id
    if [ -z "$prev_job_id" ]; then
	    prev_job_id=$(qsub $job_script)
    else
	    prev_job_id=$(qsub -W depend=afterok:$prev_job_id $job_script)
    fi
    echo "Submitted job for $p processors, $nums."
done

echo "All jobs submitted."

