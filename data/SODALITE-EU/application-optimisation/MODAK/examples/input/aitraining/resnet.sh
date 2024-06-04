#PBS -S /bin/bash
## START OF HEADER ##
#PBS -N resnet
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=40:gpus=1
#PBS -o resnet.out
#PBS -j oe
#PBS -m abe
#PBS -M kartshy@gmail.com
## END OF HEADER ##
cd $PBS_O_WORKDIR
export PATH=$PBS_O_WORKDIR:$PATH

file=enable_xla.sh
if [ -f $file ] ; then rm $file; fi
wget --no-check-certificate https://www.dropbox.com/s/e6n9yb0a5601gp2/enable_xla.sh
chmod 755 enable_xla.sh
source enable_xla.sh

singularity exec --nv  $SINGULARITY_DIR/modak_tensorflow-2.1-gpu-src.sif  ./resnet_benchmark.sh
