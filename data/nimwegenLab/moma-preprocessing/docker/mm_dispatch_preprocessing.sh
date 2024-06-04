#!/usr/bin/env bash

#
# Script dispatches the preprocessing of the dataset.
#


mm_dispatch_preprocessing(){

#MMPRE_EXIST=$(module av MMPreproc 2>&1 | grep MMPreproc | wc -l)
#if [ $MMPRE_EXIST -eq 0 ]; then printf "The module MMPreproc cannot be found. Aborting...\n" >&2; exit 1; fi

# Check arguments
if [ -n "$CAMERA_ROI_PATH" ]; then printf "WARNING: CAMERA_ROI_PATH argument is defined, but not supported. Will be ignored.\n"; fi
if [ -n "$DARK_PATH" ]; then printf "WARNING: DARK_PATH argument is defined, but not supported. Will be ignored.\n"; fi
if [ -n "$GAIN_PATH" ]; then printf "WARNING: GAIN_PATH argument is defined, but not supported. Will be ignored.\n"; fi 
if [ -n "$HOTPIXS_PATH" ]; then printf "WARNING: HOTPIXS_PATH argument is defined, but not supported. Will be ignored.\n"; fi
if [ -n "$CHANNELS_ORDER" ]; then printf "WARNING: CHANNELS_ORDER argument is defined, but not supported. Will be ignored.\n"; fi
if [ -n "$FIJI_MACRO" ]; then printf "WARNING: FIJI_MACRO argument is defined, but not supported. Will be ignored.\n"; fi

# get array length
N=${#POS_NAMES[@]}

PREPROC_DIR=$(dirname $PREPROC_DIR_TPL)
BASENAME=$(basename "$PREPROC_DIR")

# start one job per position
# using a job array is not convenient since it requires to pass indices and flip flags arrays as arguments
for (( I=0; I<N; I++ )); do
  POS_NAME=${POS_NAMES[$I]}
  CROP_ROI_PATH=$(printf $CROP_ROI_PATH_TPL $POS_NAME)
  ROTATION=${ROTATIONS[$I]}

  SCRIPT=$PREPROC_DIR/logs/slurm_${POS_NAME}.sh           # path to custom bash script
  LOG=$PREPROC_DIR/logs/slurm_${POS_NAME}.log             # path to redirect stdout (qsub command and job ID)
  S_OUT=$PREPROC_DIR/logs/slurm_${POS_NAME}_out.log  # path to redirect qsub stdout
  S_ERR=$PREPROC_DIR/logs/slurm_${POS_NAME}_err.log  # path to redirect qsub stderr

  mkdir -p $PREPROC_DIR # -p: no error if existing, make parent directories as needed
  mkdir -p $PREPROC_DIR/logs
  
#  CMD_STR="python \"$MMPRE_HOME/call_preproc_fun.py\" \
  CMD_STR="moma_preprocess\
 -i \"$RAW_PATH\"\
 -o \"$PREPROC_DIR\""
  if [ -n "$POS_NAME" ]; then CMD_STR="$CMD_STR -p $POS_NAME"; fi # append optional argument
  if [ -n "$ROTATION" ]; then CMD_STR="$CMD_STR -r $ROTATION"; fi # append optional argument
#  if [ -n "$CAMERA_ROI_PATH" ]; then CMD_STR="$CMD_STR -j $CAMERA_ROI_PATH"; fi # append optional argument
#  if [ -n "$DARK_PATH" ]; then CMD_STR="$CMD_STR -d $DARK_PATH"; fi # append optional argument
#  if [ -n "$GAIN_PATH" ]; then CMD_STR="$CMD_STR -g $GAIN_PATH"; fi # append optional argument
#  if [ -n "$HOTPIXS_PATH" ]; then CMD_STR="$CMD_STR -h $HOTPIXS_PATH"; fi # append optional argument
  if [ -n "$FLATFIELD_PATH" ]; then CMD_STR="$CMD_STR -ff $FLATFIELD_PATH"; fi # append optional argument
#  if [ -n "$CHANNELS_ORDER" ]; then CMD_STR="$CMD_STR -c $CHANNELS_ORDER"; fi # append optional argument
#  if [ -n "$FIJI_MACRO" ]; then CMD_STR="$CMD_STR -m $FIJI_MACRO"; fi # append optional argument
  if [ -n "$TMIN" ]; then CMD_STR="$CMD_STR -tmin $TMIN"; fi # append optional argument
  if [ -n "$TMAX" ]; then CMD_STR="$CMD_STR -tmax $TMAX"; fi # append optional argument
  if [ -n "$GLT" ]; then CMD_STR="$CMD_STR -glt $GLT"; fi # append optional argument
  if [ -n "$ROI_BOUNDARY_OFFSET_AT_MOTHER_CELL" ]; then CMD_STR="$CMD_STR --roi_boundary_offset_at_mother_cell $ROI_BOUNDARY_OFFSET_AT_MOTHER_CELL"; fi # append optional argument
  if [ -n "$GL_DETECTION_TEMPLATE_PATH" ]; then CMD_STR="$CMD_STR --gl_detection_template_path \"$GL_DETECTION_TEMPLATE_PATH\""; fi # append optional argument
  if [ -n "$NORMALIZATION_CONFIG_PATH" ]; then CMD_STR="$CMD_STR --normalization_config_path $NORMALIZATION_CONFIG_PATH"; fi # append optional argument
  if [ -n "$ZSLICE" ]; then CMD_STR="$CMD_STR -zslice $ZSLICE"; fi # append optional argument
  if [ -n "$NORMALIZATION_REGION_OFFSET" ]; then CMD_STR="$CMD_STR --normalization_region_offset $NORMALIZATION_REGION_OFFSET"; fi # append optional argument
  if [ -n "$FRAMES_TO_IGNORE" ]; then CMD_STR="$CMD_STR --frames_to_ignore \"$FRAMES_TO_IGNORE\""; fi # append optional argument
  if [ -n "$IMAGE_REGISTRATION_METHOD" ]; then CMD_STR="$CMD_STR --image_registration_method \"$IMAGE_REGISTRATION_METHOD\""; fi # append optional argument
  if [ -n "$FORCED_INTENSITY_NORMALIZATION_RANGE" ]; then CMD_STR="$CMD_STR --forced_intensity_normalization_range \"$FORCED_INTENSITY_NORMALIZATION_RANGE\""; fi # append optional argument

  CMD_STR="$CMD_STR -log \"$LOG\""

  if [[ "ierbert2" == "$(hostname)" ]]; then
    printf "INFO: Running on laptop. Will NOT use slurm/sbatch.\n"
    running_on_laptop=true
  fi

  anaconda_module_load_string="module load Anaconda3/5.0.1"
  if [[ "$running_on_laptop" = true ]]; then
    anaconda_module_load_string=""
  fi

  # use single quote to prevent variable evaluation
  CMD_SCRIPT="#!/bin/bash \n\n\
#SBATCH -t 1-00:00:00 \n\
#SBATCH --qos=1day \n\
#SBATCH --nodes=1 \n\
#SBATCH --ntasks=2 \n\
#SBATCH --mem-per-cpu=32G \n\
#SBATCH --export=$MODULEPATH \n\
#SBATCH -o $S_OUT \n\
#SBATCH -e $S_ERR \n\n\
\
$CMD_STR
\
\n"

printf "${CMD_SCRIPT}"

  if [[ "$running_on_laptop" = true ]]; then
    CMD_SBATCH=$SCRIPT
  else
    CMD_SBATCH="sbatch $SCRIPT"
  fi

  printf "$CMD_SCRIPT" > $SCRIPT
  chmod +x $SCRIPT

  [ -f "$S_OUT" ] && rm $S_OUT # delete if exists
  [ -f "$S_ERR" ] && rm $S_ERR
  printf "$CMD_SBATCH \n" > $LOG
  
  $CMD_SBATCH | tee -a $LOG
done
echo "Preprocessing queued... (use squeue to check the current status)"
wait

}
