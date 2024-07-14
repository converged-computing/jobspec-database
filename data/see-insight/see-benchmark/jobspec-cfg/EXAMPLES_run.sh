#!/bin/bash

export IMGNUM=$1
export PYTHONPATH="./see-segment/"

echo "Running on $HOSTNAME"

if [ ! -f imagefile.txt ]
then
    touch imagefile.txt

    for GT in ./see-segment/Image_data/Examples/*GT*
    do
       ROOTNAME=`echo $GT | sed -e "s/_GT.*$//"`
    echo `echo ${ROOTNAME}.*` $GT >> imagefile.txt
    done
fi

NUM_IMG=`cat imagefile.txt | wc -l`

MY_NUM=$(( ( $1 - 1 ) % $NUM_IMG + 1 ))

MINE=`head -n $MY_NUM imagefile.txt | tail -n 1`

continiuous=""

if [ ! "$3" == "" ]
then
    continuous="--checkpoint run"
fi


python ./see-segment/see/RunSearch.py ${continuous} --num_iter 1000000 --seed $2 $MINE
ret=$?

echo "FINNISHED RUNNING SCRIPT with error $ret"

exit $?
