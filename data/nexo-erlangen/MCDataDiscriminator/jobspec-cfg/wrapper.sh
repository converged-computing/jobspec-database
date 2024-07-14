timestamp() {
	date +"%y%m%d-%H%M"
}
timestampSEC() {
	date +"%y%m%d-%H%M-%S"
}

TIMESTAMP=$(timestamp)
TIMESTAMPSEC=$(timestampSEC)

#check ob auf GPU mit Hilfe von $HOST ?
PWD=/home/vault/capm/mppi060h/MCDataDiscriminator/
DATA="${PWD}Data/"
FOLDERRUNS="${PWD}TrainingRuns/"
CODEFOLDER=$HPC/MCDataDiscriminator/

array=( $* )
for((i=0; i<$#; i++)) ; do
	case "${array[$i]}" in
		--test) TEST="true";;
		--prepare) PREPARE="true";;
		--resume) RESUME="true";;
		--model) PARENT=${array[$i+1]};;
		-m) PARENT=${array[$i+1]};;
		--single) SINGLE="true";;
		-s) SINGLE="true";;
	esac
done 

if [[ $TEST == "true" ]] ||  [[ $PREPARE == "true" ]] ; then
	RUN=Dummy
else
	if [[ $RESUME = "true" ]] && [[ ! -z $FOLDERRUNS$PARENT ]]
	then
		if [ -d $FOLDERRUNS$PARENT ]
		then
			if [[ $SINGLE != "true" ]]
			then
				RUN=$PARENT/$TIMESTAMP
				if [ -d $FOLDERRUNS$PARENT/$TIMESTAMP ]
				then
					RUN=$PARENT/$TIMESTAMPSEC
				fi
			else
				RUN=$PARENT/
			fi
		else
			echo "Model Folder (${PARENT}) does not exist" ; exit 1
		fi
	else
		RUN=$TIMESTAMP
		if [ -d $FOLDERRUNS$TIMESTAMP ]
		then
			RUN=$TIMESTAMPSEC
		fi
	fi
fi

mkdir -p $FOLDERRUNS$RUN

echo "Run Folder:   " $FOLDERRUNS$RUN
echo

if [[ $PREPARE == "true" ]] ; then
	echo "(python $CODEFOLDER/run_cnn.py --in $DATA --runs $FOLDERRUNS --out $RUN ${@:1}) | tee $FOLDERRUNS/$RUN/log.dat"
else
	#echo "(python $CODEFOLDER/run_cnn.py --in $DATA --runs $FOLDERRUNS --out $RUN ${@:1}) | tee $FOLDERRUNS/$RUN/log.dat"
	#(python $CODEFOLDER/run_cnn.py --in $DATA --runs $FOLDERRUNS --out $RUN ${@:1}) | tee $FOLDERRUNS/$RUN/log.dat
	(python $CODEFOLDER/run_cnn.py --in $DATA --runs $FOLDERRUNS --out $RUN ${@:1})
	echo
	echo "Run Folder:   " $FOLDERRUNS$RUN
fi
