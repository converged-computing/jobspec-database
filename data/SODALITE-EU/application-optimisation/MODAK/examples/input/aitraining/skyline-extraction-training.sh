#PBS -S /bin/bash
## START OF HEADER ##
#PBS -N skyline-extraction-training
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=40:gpus=1
#PBS -l procs=40
#PBS -l nodes=ssd
#PBS -o skyline-extraction-training.out
#PBS -e skyline-extraction-training.err
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M kartshy@gmail.com
## END OF HEADER ##
cd $PBS_O_WORKDIR
export PATH=$PBS_O_WORKDIR:$PATH

## START OF TUNER ##
file=skyline-extraction-training_20200812125518_tune.sh
if [ -f $file ] ; then rm $file; fi
wget --no-check-certificate https://storage.googleapis.com/modak-bucket//modak/skyline-extraction-training_20200812125518_tune.sh
chmod 755 skyline-extraction-training_20200812125518_tune.sh

singularity exec --nv  $SINGULARITY_DIR/modak_tensorflow-2.1-gpu-src.sif skyline-extraction-training_20200812125518_tune.sh
## END OF TUNER ##

file=enable_xla.sh
if [ -f $file ] ; then rm $file; fi
wget --no-check-certificate https://www.dropbox.com/s/e6n9yb0a5601gp2/enable_xla.sh
chmod 755 enable_xla.sh
source enable_xla.sh
singularity exec --nv  $SINGULARITY_DIR/modak_tensorflow-2.1-gpu-src.sif  python3 resnet/resnet_imagenet_main.py --data_dir=/mnt/nfs/home/karthee/AI/data/tf_records/train/ -batch_size=96 --train_epochs=3 --train_steps=10 --use_synthetic_data=false
