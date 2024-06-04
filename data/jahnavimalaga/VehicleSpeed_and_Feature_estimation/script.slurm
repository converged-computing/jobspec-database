#!/bin/bash

# Author: Jahnavi Malagavalli
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 SKX nodes
#
#   * Serial Job on SKX Normal Queue *
# 
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch skx.serial.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------
#job names are unique
# frontera production q's name 
# what is the maximum limit in frontera queue, if more time it will be more time in queue to go to running  
# sbatch script.slurm
# squeue -u jahnavi
# scancel 170361
# idev -N 1 -n 1 -p rtx-dev -t 02:00:00

#CHANGE -p depending on what queue is required!! gpu-a100 has the GPUs

#SBATCH -J CompletePipleline           # Job name
#SBATCH -o CompletePipleline.out       # Name of stdout output file "%j" expands to your job's numerical job ID
#SBATCH -e CompletePipleline.err       # Name of stderr error file %j" expands to your job's numerical job ID
#SBATCH -p rtx-dev#small #gpu-a100      # Queue (partition) name
#SBATCH -N 1              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=jahnavimalagavalli@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job


# Other commands must follow all #SBATCH directives...


#module load cuda/11.4 cudnn/8.2.4 nccl/2.11.4
#source $WORK/anaconda3/etc/profile.d/conda.sh
#conda init
#conda activate good_sys

# python code2vec.py --load models/java-small/saved_model_iter15 --data data/java-small/java-small --save models2-java2-small/saved-model --test data/java-small/java-small.val.c2v
# python3 code2vec.py --load models/java14_model/saved_model_iter8.release --test data/java14m/java14m.train.c2v
#cd $WORK
 

#cd GoodSystems/Vehicle_Identification_and_speed_estimation_code/Complete_Pipeline
cd Complete_Pipeline


path_dir="test/" #"test/archive/new_videos/" #
for file in "$path_dir"/* #.{mov,mp4}
do
    file_name=$(basename "$file")
    # Remove the .py extension
    file_name_without_extension="${file_name%.*}" #"${file_name%.mp4}"
    PYTHONWARNINGS="ignore::FutureWarning" python3 main.py --name "$file_name_without_extension" --conf-thres 0.25 --source "$file" --device 0 --hide-conf --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7
    #python3 main.py --name "$file_name_without_extension" --conf-thres 0.25 --source "$file" --device 0 --hide-conf --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7

    #sleep 60
done

#python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 0 --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov8n-seg.pt --classes 1 2 3 5 7
#python3 main.py --conf-thres 0.25 --source test/20230222_115854.mp4 --device 0 --save-crop-lp --save-crop --save-vid --save-txt --strong-sort-weights weights/osnet_x0_25_msmt17.pt --yolo-weights weights/yolov7.pt --classes 1 2 3 5 7

#conda deactivate
echo "Your job has completed $SLURM_JOB_ID"
