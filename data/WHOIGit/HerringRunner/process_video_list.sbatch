#!/usr/bin/env bash

#SBATCH --job-name=framelist
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=
#SBATCH --output=slurm/%x.%A.%a.txt


echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME", ArrayTask: $SLURM_ARRAY_TASK_ID
echo "Command: $0 $@"
hostname; pwd; date


# Error if this is not a GPU node
[ -n "${CUDA_VISIBLE_DEVICES+x}" ]

module load cuda10.1/{toolkit,blas,fft,cudnn}
source herring_yolo_env/bin/activate
echo "Environment... Loaded"

# Get configuration options 
VIDEOFILE_LIST="$1"
VIDEOFILE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $VIDEOFILE_LIST)

# Get the name of the video without extension
VIDEONAME="$(basename "$VIDEOFILE" | rev | cut -d . -f 2- | rev)"

# make output directory
FRAME_OUT=detect-data/video_rawframes/"$VIDEONAME"
PROCFRAME_OUT=detect-data/video_procframes/"$VIDEONAME"
mkdir -p "$FRAME_OUT"
mkdir -p "$PROCFRAME_OUT"

echo
echo "PROCESSING VIDEO FRAMES: $VIDEOFILE"
python process_video.py \
    -v "$VIDEOFILE" \
    --progress \
    --ramdisk \
    --num-cores 1 \
    --save-original "$FRAME_OUT" \
    --save-preprocessed "$PROCFRAME_OUT" \
    --afr afr.json 
    

echo
echo "FRAMES: $(ls $FRAME_OUT | wc -l)"
echo

if [ "$#" -eq 2 ]; then
    MODELPATH=$2
    DATASET=$PROCFRAME_OUT  # may also be FRAME_OUT for NoProc models
    if [[ "$MODELPATH" == *"best.pt" ]]; then
        
        MODELNAME="$(basename "$(dirname "$(dirname "$MODELPATH")")")"
        echo "YOLO MODEL: $MODELNAME"
        cd yolov5_ultralytics
        
        time python detect.py --weights "../$MODELPATH" --source "../$DATASET" \
            --project "../detect-output/$MODELNAME" --name "$VIDEONAME" \
            --save-txt --save-null-txt --nosave --hide-labels --device $CUDA_VISIBLE_DEVICES
        cd ..
        python detect_summary.py "detect-output/$MODELNAME/$VIDEONAME/labels" \
            --outdir "detect-output/$MODELNAME" --outfile "$VIDEONAME.csv" --metadata
            
        rm -fr "detect-output/$MODELNAME/$VIDEONAME"
        
        
    else

        MODELNAME="$(basename "$MODELPATH" | rev | cut -d . -f 2- | rev)"
        echo "CLASSIFIER MODEL: $MODELNAME"
        source deactivate
        source herring_classnn_env/bin/activate
        time python pytorch_classifier/neuston_net.py RUN "$DATASET" "$MODEL" "MODELNAME__$VIDEONAME" \
             --outdir "detect-output/$MODELNAME" --outfile "$VIDEONAME.csv"
    fi
    
    # CLEANUP frame folders
    rm -fr "$FRAME_OUT"
    rm -fr "$PROCFRAME_OUT" 

else
    
    ls "$FRAME_OUT" | sort -t _ -k2 -V > detect-data/video_framelist/$VIDEONAME.rawframes
    ls "$PROCFRAME_OUT" | sort -t _ -k2 -V > detect-data/video_framelist/$VIDEONAME.procframes
    #rm -fr "$FRAME_OUT"
    #rm -fr "$PROCFRAME_OUT"

fi

echo DONE


