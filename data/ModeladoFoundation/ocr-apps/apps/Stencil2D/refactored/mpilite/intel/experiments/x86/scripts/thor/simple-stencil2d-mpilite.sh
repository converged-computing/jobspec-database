#!/bin/bash
#SBATCH --partition=XAS


### look at runme.sh file for what the six arguments below mean and do.
scale=$1
runhm=$2
MYITACIN=$3
psize=$4
EXE=$5
ph=$6

#### change the two lines below to point to your OCR and APPS root dirs. #####
export OCR_INSTALL=/home/rsalmon/dhome/fastfw/src/ocr/ocr/install
export APPS_ROOT="/home/rsalmon/dhome/fastfw/src/apps/apps"
##############################################################################

export LD_LIBRARY_PATH=${OCR_INSTALL}/lib:${APPS_ROOT}/libs/install/x86/lib:${LD_LIBRARY_PATH}:/opt/bamboo_6_0/bin/rose/lib:/opt/bamboo_6_0/bin/boost/lib:/etc/alternatives/jre_1.7.0_openjdk/lib/amd64/server

export OCR_TYPE=x86-mpi
MYOCR=$OCR_INSTALL/bin/ocrrun_x86-mpi

cd ${runhm}

# number of tiles
ntiles=(36 72 144 288 576 1152 2304)
nitr=100 #number of iterations

# fill in 128 nodes with ppn=18
nnodes=$SLURM_NNODES
np="`echo "${nnodes}*$ph"|bc`"
pow2tmp=`echo "l(${nnodes})/l(2)" |bc -l`
printf -v pow2 %.0f "$pow2tmp"

cfgfile=$runhm/generated.mpi-lite.$np.$scale.cfg
export OCR_CONFIG=$cfgfile


export I_MPI_DEBUG=3
export OCR_NODEFILE=$runhm/hfile.txt.$$
pdsh -N -w "${SLURM_NODELIST}" hostname |sort > ${OCR_NODEFILE}
#export OCRRUN_OPT_ENVKIND="SLURM"
#export OCRRUN_OPT_NUM_NODES="-n"

#################### begin itac ######################################
case $MYITACIN in
    p2p)
	source /opt/intel/itac_latest/bin/itacvars.sh -s
	export OCRRUN_ITAC=yes
	export OCRRUN_OPT_ITAC="-trace -trace-pt2pt"
	;;
    coll)
	source /opt/intel/itac_latest/bin/itacvars.sh -s
	export OCRRUN_ITAC=yes
	export OCRRUN_OPT_ITAC="-trace -trace-collectives"
	;;
    all)
	source /opt/intel/itac_latest/bin/itacvars.sh -s
	export OCRRUN_ITAC=yes
	export VT_CONFIG="/home/rsalmon/dhome/fastfw/src/apps/apps/CoMD/refactored/mpilite/vtconfig.conf"
	#export VT_PCTRACE=6
	export OCRRUN_OPT_ITAC="-trace"
	;;
    none)
	#no itac.
esac
#################### end itac ######################################

case $scale in
    strong)
	#strong scaling test. Problem size stays constant.
	case $psize in
	    small)
		gsize=23170
		;;
	    medium)
		gsize=32768  # Medium problem size
		;;
	    large)
		gsize=46340  # large problem size
	esac

	i=`echo "${pow2}-1"|bc`
	lname=$scale.np.$np.log

	export CONFIG_NUM_THREADS=`echo "$ph+2" |bc`
	export OCR_NUM_NODES=${nnodes}
	${OCR_INSTALL}/share/ocr/scripts/Configs/config-generator.py --remove-destination --threads $CONFIG_NUM_THREADS --output $cfgfile --target mpi --guid COUNTED_MAP --guid LABELED --scheduler LEGACY --dequetype LOCKED_DEQUE

	echo "##s########################## np=$np #############################"
	echo "    problem size           Num tiles "
	echo " ${gsize} x ${gsize}          "
	echo "	$MYOCR ${EXE} -r $np -t 110 ${nitr} ${gsize} "

	$MYOCR ${EXE} -r $np -t 110 ${nitr} ${gsize}

	;;
    weak)
	#weak scaling test. Problem size increases with np. ratio of work stays the same
	case $psize in
	    small)
		gsize=(9216 9216 18432 18432 36864 36864 73728)
		;;
	    medium)
		gsize=(18432 18432 36864 36864 73728 73728 147456)  # Medium problem size
		;;
	    large)
		gsize=(36864 36864 73728 73728 147456 147456 294912) # Large problem size
	esac

	i=`echo "${pow2}-1"|bc`
	lname=$scale.np.$np.log

	export CONFIG_NUM_THREADS=`echo "$ph+2" |bc`
	export OCR_NUM_NODES=${nnodes}
	${OCR_INSTALL}/share/ocr/scripts/Configs/config-generator.py --remove-destination --threads $CONFIG_NUM_THREADS --output $cfgfile --target mpi --guid COUNTED_MAP --guid LABELED  --scheduler LEGACY --dequetype LOCKED_DEQUE
	echo "##w########################## np=$np #############################"
	echo "    problem size          Rank config"
	echo " ${gsize[$i]} x ${gsize[$i]}             ${ntiles[$i]}"
	echo "	$MYOCR ${EXE} -r $np -t 110 ${nitr} ${gsize[$i]}"

	$MYOCR ${EXE} -r $np -t 110 ${nitr} ${gsize[$i]}


esac


