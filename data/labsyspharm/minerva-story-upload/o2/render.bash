#!/bin/bash
#SBATCH -c 1
#SBATCH -p short
#SBATCH --mem 1g
#SBATCH -t 0:30:00
#SBATCH --array=0-14
module load conda3
eval "$(conda shell.bash hook)"
conda activate minerva-author
which python

DATE="2023-10-11"
TITLE="author-20XX"
SAMPLES=(
"Sample1"
"Sample2"
"Sample3"
"Sample4"
"Sample5"
"Sample6"
"Sample7"
"Sample8"
"Sample9"
"Sample10"
"Sample11"
"Sample12"
"Sample13"
"Sample14"
"Sample15"
)
BUCKET="www.cycif.org"
SAMPLE="${SAMPLES[$SLURM_ARRAY_TASK_ID]}"
SCRATCH="/n/scratch3/users/${USER:0:1}/${USER}"
SCRIPT="/home/${USER}/minerva-author/src/render.py --force"
JSON="/home/${USER}/${DATE}/sources/${SAMPLE}.story.json"
if [ ! -f $JSON ]; then
    JSON="/home/${USER}/${DATE}/sources/default.story.json"
fi
echo "Rendering ${SAMPLE}"

create_story () {
  STORY_IMG="${SCRATCH}/${DATE}/${SAMPLE}.ome.tif"
  STORY_URL="https://s3.amazonaws.com/${BUCKET}/${TITLE}/${SAMPLE}"
  STORY_DIR="${SCRATCH}/${DATE}/${SAMPLE}"

  CMD="$SCRIPT $STORY_IMG $JSON $STORY_DIR --url $STORY_URL"

  echo "python $CMD"
  python $CMD
}

create_story
