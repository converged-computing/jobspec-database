#!/bin/bash -l
#PBS -l walltime=10:00:00
#PBS -l mem=8gb
#PBS -m abe
#PBS -M gbonaert@ulb.ac.be
#PBS -q gpu
#PBS -l nodes=1:ppn=4:gpus=1:gpgpu

module purge
module load GCCcore/6.4.0
module load Szip/2.1.1-GCCcore-6.4.0
module load binutils/2.28-GCCcore-6.4.0
module load HDF5/1.10.1-foss-2017b
module load GCC/6.4.0-2.28
module load pkg-config/0.29.2-GCCcore-6.4.0
module load numactl/2.0.11-GCCcore-6.4.0
module load pkgconfig/1.2.2-foss-2017b-Python-3.6.3
module load hwloc/1.11.7-GCCcore-6.4.0
module load h5py/2.7.1-foss-2017b-Python-3.6.3
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
module load libyaml/0.1.7-GCCcore-6.4.0
module load OpenBLAS/0.2.20-GCC-6.4.0-2.28
module load PyYAML/3.12-foss-2017b-Python-3.6.3
module load gompi/2017b
module load Keras/2.1.3-foss-2017b-Python-3.6.3
module load FFTW/3.3.6-gompi-2017b
module load expat/2.2.4-GCCcore-6.4.0
module load ScaLAPACK/2.0.2-gompi-2017b-OpenBLAS-0.2.20
module load fontconfig/2.12.4-GCCcore-6.4.0
module load foss/2017b
module load X11/20171023-GCCcore-6.4.0
module load bzip2/1.0.6-GCCcore-6.4.0
module load Tk/8.6.7-foss-2017b
module load zlib/1.2.11-GCCcore-6.4.0
module load Tkinter/3.6.3-foss-2017b-Python-3.6.3
module load ncurses/6.0-GCCcore-6.4.0
module load libpng/1.6.32-GCCcore-6.4.0
module load libreadline/7.0-GCCcore-6.4.0
module load freetype/2.8-GCCcore-6.4.0
module load Tcl/8.6.7-GCCcore-6.4.0
module load scikit-learn/0.19.1-foss-2017b-Python-3.6.3
module load SQLite/3.20.1-GCCcore-6.4.0
module load Qhull/2015.2-foss-2017b
module load GMP/6.1.2-GCCcore-6.4.0
module unload matplotlib
module load matplotlib/2.1.1-foss-2017b-Python-3.6.3
module load XZ/5.2.3-GCCcore-6.4.0
module load NASM/2.13.01-GCCcore-6.4.0
module load libffi/3.2.1-GCCcore-6.4.0
module load libjpeg-turbo/1.5.2-GCCcore-6.4.0
module load Python/3.6.3-foss-2017b
module load LibTIFF/4.0.9-foss-2017b
module load CUDA/9.1.85
module load Pillow/4.3.0-foss-2017b-Python-3.6.3
module load cuDNN/7.0.5-CUDA-9.1.85
module load scikit-image/0.13.1-foss-2017b-Python-3.6.3
module load TensorFlow/1.5.0-foss-2017b-Python-3.6.3-CUDA-9.1.85


cd $PBS_O_WORKDIR
echo submit directory: $PWD
echo jobid: $PBS_JOBID
echo hostname: $HOSTNAME
date
echo --- Start Job ---
python optimizer.py
#python main.py
echo ---- End Job ----
date

