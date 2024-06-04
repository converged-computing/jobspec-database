#!/usr/bin/bash --login
#SBATCH --mem=16gb --gres=gpu:k80:1 -A general  --time=2:00:00


### Notes
# the time above is set to 2hrs, but when calling this batch file, 
# you can estimate the needed run time as 2 sec * number of photos
# and set with --time parameter.  

##### note when running this you must set INPUT_FOLDER like so: 

USAGE='PHOTOFOLDER=path/to/photos; sbatch --export=INPUT_FOLDER=$PHOTOFOLDER --job-name=detector-$(basename $PHOTOFOLDER) run_detector.sb'

# modify this for your own python environment location
if [ -z ${PYTHON_FOLDER} ]; then     
    PYTHON_FOLDER=$HOME/python37tf/
    echo "using python folder $PYTHON_FOLDER"
fi

if [ -z "$INPUT_FOLDER" ]; then 
    echo "INPUT_FOLDER variable is required, example slurm command: "
    echo $USAGE
    exit 1
fi

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Folder not found : $INPUT_FOLDER"
    exit 1
fi

OUTPUT_FOLDER=output/`basename ${INPUT_FOLDER}`
# load HPCC modules needed, but use our own python environment
ml purge
ml  GNU/8.2.0-2.31.1 Python/3.7.2  CUDA/10.1.105 cuDNN/7.6.4.38

# activate python environment that has Tensorflow installed
source $PYTHON_FOLDER/bin/activate

python run_detector.py  $INPUT_FOLDER $OUTPUT_FOLDER

