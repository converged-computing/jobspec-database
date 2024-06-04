#!/bin/bash

# create pbs file

jobName='track'
#jobDict='/scratch/jiadeng_fluxoe/mzwang/naming/src'
#jobDict='/home/mzwang/workspace/fast-rcnn'
#jobDict='/home/mzwang/libs/edges-master'
jobDict='/scratch/jiadeng_fluxoe/mzwang/naming/src/facenet/src/align'
pbsFile='feat.pbs'
node=1
ppn=1
mem='8gb'
flux='fluxoe'
runtime='03:00:00:00'
logfile='detect.log'
i=1
#for i in `seq 1 9`;
#for dir in /scratch/jiadeng_fluxoe/mzwang/mvqa/MovieQA_benchmark/story/video_clips/tt*; do
while IFS='' read -r line || [[ -n "$line" ]]; do
#for vid in /scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/full_videos/clips/pirate/*; do
	fn=$pbsFile${i}
	echo -e '####  PBS preamble\n' > $fn
	echo '#PBS -N '$jobName$line >> $fn
	echo -e '#PBS -M mzwang@umich.edu\n#PBS -m abe\n' >> $fn
	echo '#PBS -A jiadeng_'$flux >> $fn
	echo -e '#PBS -l qos=flux\n#PBS -q '$flux'\n' >> $fn
	echo '#PBS -l nodes='${node}':ppn='${ppn}':gpus=1,pmem='$mem >> $fn
	echo '#PBS -l nodes='${node}':ppn='${ppn}',pmem='$mem >> $fn
	echo '#PBS -l walltime='$runtime >> $fn
	echo -e '#PBS -j oe\n#PBS -V' >> $fn
	echo '#PBS -d '$jobDict >> $fn
	echo -e '\n####  End PBS preamble\n'>> $fn
	echo -e 'if [ -s "$PBS_NODEFILE" ] ; then\n\techo "Running on"\n\tcat $PBS_NODEFILE\nfi' >> $fn
	echo -e '\nif [ -d "$PBS_O_WORKDIR" ] ; then\n\tcd $PBS_O_WORKDIR\n\techo "Running from $PBS_O_WORKDIR"\nfi\n' >> $fn
	echo '#  Put your job commands after this line' >> $fn
	#echo 'matlab -nodisplay -nosplash -r "mergepvall('${i}')" > '$logfile${i} >> $fn
	#echo 'python generate_bbox.py '$i' 10 > '$logfile${i} >> $fn
	#echo 'python width_all.py '$i' > '$logfile${i} >> $fn
	#echo 'th train.lua -lrd 0 -lr '$i' > '$logfile${i} >> $fn
	#echo 'th test.lua -lr '$i' > '$logfile'_test'${i} >> $fn
	#echo 'th test_sp.lua -lamda 0.'$i' -lr '$i' -file output/matching_visual_cons_250_0.000000_0.000100_0.000200_3_2_3.asc' >> $fn
	#echo 'th test_sp.lua -lamda 0.'$i' -lr '$i' -file output/cca.asc' >> $fn
	#echo 'cd ~/dataset/Flickr30kEntities/Flickr30kPhraseLocalizationEval/' >> $fn
	#echo 'matlab -nodisplay -nosplash -r "runEval_arg('$i')"' >> $fn
	#echo 'source ~/workspace/python-virtual-environments/mvqa/bin/activate' >> $fn
	#echo 'module load ffmpeg image-libraries opencv cuda cudnn' >> $fn	
	#echo 'python cropimg.py '$line >> $fn
	#echo 'module load ffmpeg image-libraries opencv cuda cudnn mkl torch' >> $fn 
	#echo 'th extract_feat_video.lua -vid '$vid >> $fn
	echo 'module load cuda/8.0.44' >> $fn
	echo  'module load ffmpeg image-libraries opencv/2.4.13' >> $fn
	echo 'module load mkl/11.3.3' >> $fn
	echo 'module load hdf5/1.8.16/gcc/5.4.0' >> $fn
	echo 'module load cudnn/8.0-v5.1' >> $fn
	echo 'source ~/workspace/python-virtual-environments/mvqa/bin/activate' >> $fn
	echo 'python track.py /scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/video_clips/'$line >> $fn
	qsub $fn
	i=$((i+1))	
done < "$1"
