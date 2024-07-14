#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=16,walltime=30:00:00

## modules
#echo "Loading modules"
#module unload matlab
#module load matlab/2017a
#module unload python
#module load dipy/dev
#echo "Finished loading modules"
#export PYTHONPATH=/N/u/brlife/git/nibabel:$PYTHONPATH

echo "Check the inputs subject id"
subjID=`jq -r '._inputs[0].meta.subject' config.json`
stat_life_id=`jq -r '._inputs[0].meta.subject' config.json`
stat_t1_id=`jq -r '._inputs[1].meta.subject' config.json`
mov_afq_id=`jq -r '._inputs[2].meta.subject' config.json`
mov_life_id=`jq -r '._inputs[3].meta.subject' config.json`
mov_t1_id=`jq -r '._inputs[4].meta.subject' config.json`
if [ $mov_life_id == $mov_t1_id -a $mov_life_id == $mov_afq_id -a $stat_life_id == $stat_t1_id ]; then
	echo "Inputs subject IDs correctly inserted"
else
	echo "Inputs subject IDs incorrectly inserted. Check them again."
	exit 1
fi

echo "Copying moving and static tractograms"
moving=`jq -r '.tractogram_moving' config.json`
static=`jq -r '.tractogram_static' config.json`
if [[ $moving == *.tck ]]; then
	cp $moving ./$mov_life_id'_track.tck';
	cp $static ./$stat_life_id'_track.tck';
else
	cp $moving ./;
	cp $static ./;
fi

echo "Tractogram conversion to trk"
if [ -f $mov_life_id'_track.tck' -a $stat_life_id'_track.tck' ];then
	echo "Input in tck format. Convert it to trk."
	mov_t1=`jq -r '.t1_moving' config.json`
	stat_t1=`jq -r '.t1_static' config.json`
	#python tck2trk.py $mov_t1 $mov_life_id'_track.tck';
	#python tck2trk.py $stat_t1 $stat_life_id'_track.tck';
        singularity exec -e docker://brainlife/dipy:0.14 bash -c "python ./tck2trk.py $mov_t1 ${mov_life_id}_track.tck && python ./tck2trk.py $stat_t1 ${stat_life_id}_track.tck"
else
	echo "Input in fe format. Convert it to trk."
	#matlab -nosplash -nodisplay -r lifeConverter
        singularity exec docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/lifeConverter

	mv life_moving_output.trk $mov_life_id'_track.trk';
	mv life_static_output.trk $stat_life_id'_track.trk';
	ret=$?
	if [ ! $ret -eq 0 ]; then
		echo "Tractogram conversion failed"
		echo $ret > finished
		exit $ret
	fi
fi

echo "AFQ conversion to trk"
#matlab -nosplash -nodisplay -r "afqConverter1()"
singularity exec docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/afqConverter1
ret=$?
	if [ ! $ret -eq 0 ]; then
		echo "AFQ conversion failed"
		echo $ret > finished
		exit $ret
	fi


echo "Running LAP single example"
mkdir tracts_tck;
singularity exec -e docker://brainlife/dipy:0.14 python ./execute_single_lap.py -mov_ID ${mov_life_id} -stat_ID ${stat_life_id} -list tract_name_list.txt
ret=$?
if [ ! $ret -eq 0 ]; then
    echo "LAP single example failed"
    exit $ret
fi

echo "Build partial tractogram"
output_filename=${subjID}'_var-partial_tract_E'${mov_afq_id}'.tck';
#python build_partial_tractogram.py -tracts_tck_dir 'tracts_tck' -out ${output_filename};
singularity exec -e docker://brainlife/dipy:0.14 python ./build_partial_tractogram.py -tracts_tck_dir tracts_tck -out $output_filename
if [ -f ${output_filename} ]; then 
    echo "Partial tractogram built."
else 
	echo "Partial tractogram missing."
	exit 1
fi

echo "Build a wmc structure"
stat_sub=\'$subjID\'
mov_sub=\'$mov_afq_id\'
#matlab -nosplash -nodisplay -r "build_wmc_structure($stat_sub, $mov_sub)";
singularity exec docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/build_wmc_structure $subjID $mov_afq_id
if [ -f 'output.mat' ]; then 
    echo "WMC structure created."
else 
	echo "WMC structure missing."
	exit 1
fi

echo "Complete"
