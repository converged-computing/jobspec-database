#!/bin/bash
#BSUB -n 1
#BSUB -W 24:00
#BSUB -J %J
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu
#BSUB -R "select[rtx2080||gtx1080||p100]"
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -R "rusage[mem=32GB]"
#-------------------------------------
# REMEMBER TO PASS $SRC and $DST TO THE JOB COMMAND:
# bsub -env "PARENT_DIRECTORY=$PARENT_DIRECTORY, REMOTE=$REMOTE" < predict.sh
# Parent dir where deployments dirs are stored
# PARENT_DIRECTORY="$PARENT_DIRECTORY"  # <<<<<<<<<<<<<<<<<<<<<<
# REMOTE="$REMOTE"  # <<<<<<<<<<<<<<<<<<<<<<
#-------------------------------------
module unload python
module load conda
module load cuda/10.2
module load rclone
conda activate yolov5
#-------------------------------------
if ! [ -f 'md_v5a.0.0.pt' ]; then
  VERS='v5.0/md_v5a.0.0.pt'
  wget "https://github.com/microsoft/CameraTraps/releases/download/${VERS}" \
    -O 'md_v5a.0.0.pt'
fi
#-------------------------------------
DIRECTORIES=$(
  python -c "from glob import glob; \
    print(' '.join(
    [f'\"{x}\"' for x in glob('$PARENT_DIRECTORY/*') if '-results' not in x
    and '-with-detection-original' not in x and 'archived' not in x.lower()]))"
)
#-------------------------------------
for DIRECTORY in ${DIRECTORIES[*]}; do
  python yolov5/detect.py \
    --weights 'md_v5a.0.0.pt' \
    --source "$DIRECTORY" \
    --device 'cpu' \
    --save-txt --save-conf \
    --project "${DIRECTORY}-results" \
    --name ''
done
#-------------------------------------
for DIRECTORY in ${DIRECTORIES[*]}; do
  python filter_results.py "$DIRECTORY" || exit 1
done
#-------------------------------------
for DIRECTORY in ${DIRECTORIES[*]}; do
  rclone copy "${DIRECTORY}-with-detection-original" \
    "$REMOTE":"${PARENT_DIRECTORY}/${DIRECTORY}-with-detection-original" \
    --drive-shared-with-me -P --stats-one-line --transfers 100
done
#-------------------------------------
mkdir -p "${PARENT_DIRECTORY}/archived"
for DIRECTORY in ${DIRECTORIES[*]}; do
  mv "${DIRECTORY}" "${PARENT_DIRECTORY}/archived"
  mv "${DIRECTORY}-results" "${PARENT_DIRECTORY}/archived"
  mv "${DIRECTORY}-with-detection-original" "${PARENT_DIRECTORY}/archived"
done
