#!/bin/sh
#SBATCH --partition=GPUQ 
#SBATCH --account=ie-idi
#SBATCH --mem=192GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=4 
#SBATCH --job-name=Unet_TD
#SBATCH --time=50:00:00 
#SBATCH --export=ALL 
#SBATCH --mail-user=Bendik_haugen@hotmail.com
#SBATCH --gres=gpu:1
nvidia-smi
nvidia-smi nvlink -s
nvidia-smi topo -m
module purge
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Anaconda3/2020.07
module load fosscuda/2020b 
module laod TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1

__conda_setup="$('/cluster/apps/eb/software/Anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
   if [ -f "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
      . "/cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh"
   else
      export PATH="/cluster/apps/eb/software/Anaconda3/2020.07/bin:$PATH"
   fi
fi
unset __conda_setup

conda activate /cluster/home/bendihh/.conda/envs/sleap



# $$1 - batch size 
# $2 - tracker: "simple" | $2
# $3  - tracking_window
# 
# Unet TD 
sleap-track "video/channel1.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_td_channel1.predictions.slp   
sleap-track "video/channel1.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_td_channel1.predictions.slp   
sleap-track "video/channel1.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_td_channel1.predictions.slp   
sleap-track "video/channel2.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_td_channel2.predictions.slp  
sleap-track "video/channel2.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_td_channel2.predictions.slp  
sleap-track "video/channel2.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_td_channel2.predictions.slp  
sleap-track "video/channel3.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_td_channel3.predictions.slp  
sleap-track "video/channel3.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_td_channel3.predictions.slp  
sleap-track "video/channel3.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_td_channel3.predictions.slp  
sleap-track "video/channel4.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m Unet_td/models/Initial/Centered_instance -m Unet_td/models/Initial/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_td_channel4.predictions.slp

## reduced filters
sleap-track "video/channel1.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_td_channel1.predictions.slp   
sleap-track "video/channel1.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_td_channel1.predictions.slp   
sleap-track "video/channel1.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_td_channel1.predictions.slp   
sleap-track "video/channel2.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_td_channel2.predictions.slp  
sleap-track "video/channel2.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_td_channel2.predictions.slp  
sleap-track "video/channel2.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_td_channel2.predictions.slp  
sleap-track "video/channel3.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_td_channel3.predictions.slp  
sleap-track "video/channel3.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_td_channel3.predictions.slp  
sleap-track "video/channel3.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_td_channel3.predictions.slp  
sleap-track "video/channel4.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m Unet_td/models/Half_Filter/Centered_instance -m Unet_td/models/Half_Filter/Centroid --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_td_channel4.predictions.slp



# Unet BU
sleap-track "video/channel1.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_bu_channel1.predictions.slp  
sleap-track "video/channel1.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_bu_channel1.predictions.slp  
sleap-track "video/channel1.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o unet_bu_channel1.predictions.slp  
sleap-track "video/channel2.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_bu_channel2.predictions.slp 
sleap-track "video/channel2.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_bu_channel2.predictions.slp 
sleap-track "video/channel2.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o unet_bu_channel2.predictions.slp 
sleap-track "video/channel3.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_bu_channel3.predictions.slp 
sleap-track "video/channel3.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_bu_channel3.predictions.slp 
sleap-track "video/channel3.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o unet_bu_channel3.predictions.slp 
sleap-track "video/channel4.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_bu_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_bu_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o unet_bu_channel4.predictions.slp

# ResNet TD
sleap-track "video/channel1.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o resnet_td_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o resnet_td_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o resnet_td_channel1.predictions.slp
sleap-track "video/channel2.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o resnet_td_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o resnet_td_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o resnet_td_channel2.predictions.slp
sleap-track "video/channel3.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o resnet_td_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o resnet_td_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o resnet_td_channel3.predictions.slp
sleap-track "video/channel4.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o resnet_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o resnet_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m resnet_td/Initial/Centroid -m resnet_td/Initial/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o resnet_td_channel4.predictions.slp

# ResNet BU 
sleap-track "video/channel1.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o resnet_bu_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o resnet_bu_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o resnet_bu_channel1.predictions.slp
sleap-track "video/channel2.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o resnet_bu_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o resnet_bu_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o resnet_bu_channel2.predictions.slp
sleap-track "video/channel3.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o resnet_bu_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o resnet_bu_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o resnet_bu_channel3.predictions.slp
sleap-track "video/channel4.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o resnet_bu_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o resnet_bu_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m restnet_bu/Initial/multi_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o resnet_bu_channel4.predictions.slp

# LEAP TD
sleap-track "video/channel1.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_td_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_td_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_td_channel1.predictions.slp
sleap-track "video/channel2.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_td_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_td_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_td_channel2.predictions.slp
sleap-track "video/channel3.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_td_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_td_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_td_channel3.predictions.slp
sleap-track "video/channel4.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_td/models/Initial/centroid -m leap_td/models/Initial/centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_td_channel4.predictions.slp

## reduced filters

sleap-track "video/channel1.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_td_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_td_channel1.predictions.slp
sleap-track "video/channel1.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_td_channel1.predictions.slp
sleap-track "video/channel2.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_td_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_td_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_td_channel2.predictions.slp
sleap-track "video/channel3.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_td_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_td_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_td_channel3.predictions.slp
sleap-track "video/channel4.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_td_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_td/models/Quarter_filters/Centroid -m leap_td/models/Quarter_filters/Centered_instance --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_td_channel4.predictions.slp


# LEAP BU
sleap-track "video/channel1.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_bu_channel1.predictions.slp 
sleap-track "video/channel1.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_bu_channel1.predictions.slp 
sleap-track "video/channel1.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 7  -o leap_bu_channel1.predictions.slp 
sleap-track "video/channel2.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_bu_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_bu_channel2.predictions.slp
sleap-track "video/channel2.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 14 -o leap_bu_channel2.predictions.slp
sleap-track "video/channel3.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_bu_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_bu_channel3.predictions.slp
sleap-track "video/channel3.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 21 -o leap_bu_channel3.predictions.slp
sleap-track "video/channel4.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_bu_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_bu_channel4.predictions.slp
sleap-track "video/channel4.mp4" -m leap_bu/models/Initial --frames 0-200 --batch_size $1 --tracing.track_window $3 --tracking.tracker $2 --tracking.clean_instance_count 28 -o leap_bu_channel4.predictions.slp
