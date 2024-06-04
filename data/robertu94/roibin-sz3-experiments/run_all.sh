#!/usr/bin/env bash
#PBS -N roibin_sz
#PBS -l select=10:ncpus=40:mem=370gb:interconnect=hdr:mpiprocs=40,walltime=24:00:00
#PBS -m abe
#PBS -M robertu@clemson.edu
#PBS -j oe

workdir=/home/robertu/scratch/roibin-sz3-experiments
cd $workdir

# performance assessment
source /home/robertu/git/spack/share/spack/setup-env.sh
spack env activate .
module load openmpi/4.0.5-gcc/8.4.1-ucx

echo overview==
# for cxi_file in "/scratch1/robertu/chuck/cxic00318_0123_0.cxi" "/scratch1/robertu/chuck/cxic0415_0101.cxi"
# do 
# for chunk_size in 1 8
# do
# for replica in `seq 3`
# do
# for config in share/*.json
# do
#   echo "chunk_size=$chunk_size replica=$replica config=$config filename=$cxi_file"
#   mpiexec ./build/roibin_test -c $chunk_size -f "$cxi_file" -p "$config"
# done
# done
# done
# done

echo smallscale===
# small-scale quality assessment
# chunk_size=1
# cxi_file=/scratch1/robertu/chuck/cxic00318_0123_0.cxi
# for config in share/roibin_sz3.json share/roibin_sz.json #share/roibin_zfp.json share/roibin_mgard.json share/sz-*.json share/roibin_blosc-9.json share/sz.json share/sz3-*.json share/mgard-*.json share/zfp-*.json
# do
#   echo "chunk_size=$chunk_size replica=$replica config=$config filename=$cxi_file"
# 	mpiexec ./build/roibin_test -c $chunk_size -f "$cxi_file" -o "$cxi_file.$(basename $config)" -p "$config"
# done

echo fullscale===
# full-scale quality assessment
# config=./share/roibin_sz.json
# chunk_size=1
# replica=1
# for cxi_file in /scratch1/robertu/chuck/full_eval/cxic00318_0123_*.cxi
# do
#   echo "chunk_size=$chunk_size replica=$replica config=$config filename=$cxi_file"
# 	mpiexec ./build/roibin_test -c $chunk_size -f "$cxi_file" -o "$cxi_file.$(basename $config)" -p "$config"
# done

echo scalability===
# replica=1
# config=./share/roibin_sz.json
# chunk_size=1
# cxi_file=/scratch1/robertu/chuck/cxic00318_0123_0.cxi
# for procs in `seq 30 30 150`
# do
#   echo "chunk_size=$chunk_size replica=$replica config=$config filename=$cxi_file"
# 	mpiexec -np $procs ./build/roibin_test -c $chunk_size -f "$cxi_file" -p "$config"
# done
# echo 

echo opt===
# replica=1
# chunk_size=1
# cxi_file=/scratch1/robertu/chuck/cxic00318_0123_0.cxi
# for config in share/opt/*.json
# do
#   echo "chunk_size=$chunk_size replica=$replica config=$config filename=$cxi_file"
# 	mpiexec ./build/roibin_test -c $chunk_size -f "$cxi_file" -o "$cxi_file.$(basename $config)" -p "$config"
# done

echo tune===
# replica=1
# cxi_file=/scratch1/robertu/chuck/cxic00318_0123_0.cxi
# for chunk_size in 1 16 32 64
# do
# for config in share/tune/*.json
# do
# 	if grep -q "untune" <<<"$config"; then
# 		procs=40
# 	else
# 		procs=30
# 	fi
#   echo "chunk_size=$chunk_size replica=$replica config=$config filename=$cxi_file" procs=$procs
# 	mpiexec -np $procs ./build/roibin_test -c $chunk_size -f "$cxi_file" -p "$config"
# done
# done

echo full_table2===
chunk_size=1
replica=1
for cxi_file in /scratch1/robertu/roibin_full/r0096/cxic0415_0096.cxi /scratch1/robertu/roibin_full/r0040/cxic0415_0040.cxi /scratch1/robertu/roibin_full/r0101/cxic0415_0101.cxi
do
for config in share/table2/*.json
do
  echo "chunk_size=$chunk_size replica=$replica config=$config filename=$cxi_file"
	mpiexec ./build/roibin_test -c $chunk_size -f "$cxi_file" -o "$cxi_file.$(basename $config)" -p "$config"
done
done
