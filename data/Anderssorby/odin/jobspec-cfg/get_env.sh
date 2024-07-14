#!/usr/bin/env bash


# Set env
if [ -f /fefs/opt/dgx/env_set/aip-gpinfo.sh ]; then
 . /fefs/opt/dgx/env_set/aip-gpinfo.sh
else
 echo "Assuming test"
fi

# print date and time
echo Time is `date`
echo Directory is `pwd`
python --version

export PROJECT=CompressedDNN
if [ ! -d odin ] && [ ! -d ${PROJECT} ]; then
 echo "Project $PROJECT is ill defined."
 echo "\$HOME is $HOME"
 exit
else
 cd ${PROJECT}
fi
