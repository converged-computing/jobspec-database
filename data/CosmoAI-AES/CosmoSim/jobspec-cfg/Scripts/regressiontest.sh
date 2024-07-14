#! /bin/sh

echo Running: $0

if test x$COSMOSIM_DEVELOPING = xyes
then
   cmake --build build || exit 10
fi

dir=$1
test $dir || dir=Test/`date "+%Y%m%d"`
mkdir -p $dir

baseline=$2
test $baseline || baseline=Test/baseline20230704

F=mask
# F="mask centred reflines"
# centred and reflines are done in python and are not needed
# when validating C++ code


for flag in $F plain 
do
   mkdir -p $dir/$flag
   mkdir -p Test/diff/$flag
   mkdir -p Test/montage/$flag
done

python3 CosmoSimPy/datagen.py --directory="$dir"/plain \
   --csvfile Datasets/debug.csv  || exit 2

python3 CosmoSimPy/datagen.py --mask --directory="$dir"/mask \
   --csvfile Datasets/debug.csv  || exit 3

### python3 CosmoSimPy/datagen.py --reflines --centred --directory="$dir"/centred \
###    --csvfile Datasets/debug.csv  || exit 4
### python3 CosmoSimPy/datagen.py --reflines --directory="$dir"/reflines \
###    --csvfile Datasets/debug.csv  || exit 5

if test x$OSTYPE = xcygwin -o x$OSTYPE = xmsys
then
   if which magick
   then 
      CONVERT="magick convert"
   fi
elif which convert
then
    CONVERT="convert"
    convert --version
elif which magick
then
    CONVERT="magick convert"
    magick --version
fi


if test ! -d $baseline 
then 
   echo $baseline does not exist 
   exit 6 
elif test -z "$CONVERT"
then
     echo ImageMagick is not installed 
     exit 7 
else
    for flag in plain $F
    do
       echo $flag
       python3 CosmoSimPy/compare.py --diff Test/diff/$flag \
          $baseline/$flag $dir/$flag --masked
       COMPAREEXIT=$?
       echo python script exited with error code $COMPAREEXIT
       for f in Test/diff/$flag/*
       do
          ff=`basename $f`
          $CONVERT $baseline/$flag/$ff Test/diff/$flag/$ff $dir/$flag/$ff  \
                  +append Test/montage/$flag/$ff
       done
       if test $COMPAREEXIT -gt 0
       then
	  exit $COMPAREEXIT
       fi
    done

    echo $F
fi
