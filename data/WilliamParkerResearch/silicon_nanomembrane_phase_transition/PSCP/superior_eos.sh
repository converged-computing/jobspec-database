#!/bin/bash
#PBS -r n
#PBS -m n
#PBS -N EOS_4-atom_1B
#PBS -V
#PBS -o _scheduler-stdout.txt
#PBS -e _scheduler-stderr.txt
#PBS -q pleiades
#PBS -l walltime=372:00:00
#PBS -l select=9:mpiprocs=4

cd "$PBS_O_WORKDIR"

NCOREPERNODE=4
NPOOL=9
NPROC=`expr $NPOOL \* $NCOREPERNODE`

BIN_DIR=/usr/src/qe-6.4.1/bin
PW=$BIN_DIR/pw.x

source /opt/intel/bin/compilervars.sh intel64
source /opt/intel/mkl/bin/mklvars.sh intel64

N_ML=$(pwd | rev | cut -f4 -d'/' - | rev | cut -f1 -d'-' -)
P_DIRECTORY=$(pwd | rev | cut -f2 -d'/' - | rev)

if [ "$P_DIRECTORY" = "diamond" ]
then
        PHASE=Fd-3m
fi

if [ "$P_DIRECTORY" = "betasn" ]
then
        PHASE=I4_1amd
fi


PREFIX=Si.$PHASE\_$N_ML
RUN_TYPE=relax
INPUT=$PREFIX\.$RUN_TYPE\.in
OUTPUT=`echo $INPUT | sed 's/\.in/\.out/'`
nat=`grep "nat" $INPUT |awk '{print $3}'`
natp=$(echo "$nat+1" | bc -l)





###=======================First Set of deviations========================###
initial=0.00
step=0.05
final=0.50
run_sign=b		#b runs both positive and negative values in sequence, p runs only positive, and n runs only negative



#=====================================
				     #
				     #
if [[ "$run_sign" == "p" ]];	     #
then				     #
	run_sign_arr=$(echo "p")     #
elif [[ "$run_sign" == "n" ]];	     #
then				     #
        run_sign_arr=$(echo "n")     #
else				     #
        run_sign_arr=$(echo "p n")   #
fi				     #
				     #
				     #
#=====================================


dec_places=$(echo "$step"| awk -F. '{print $2}' | wc -m | awk '{printf "%i",$1-1}')


for strain_mag in `seq $initial $step $final`
do
	for sign in $run_sign_arr
	do

	#==============================================================================================

	        previous_strain_mag=$(echo "$strain_mag $step"| awk -v d="$dec_places" '{printf("%.*f",d,$1-$2)}')


		if [[ "$sign" == "p" ]];	
		then
			strain=$strain_mag
			previous_strain=$previous_strain_mag
		elif [[ "$sign" == "n" ]];
                then
		        strain=-$strain_mag
			previous_strain=-$previous_strain_mag
		fi


		if [[ "$sign" == "n" ]] && (( $(echo "$strain_mag" == "0" | bc -l) ));
		then
			break
		fi


		if [[ "$sign" == "n" ]] && (( $(echo "$strain_mag" == "$step" | bc -l) ));
                then
			previous_strain=$previous_strain_mag
		fi                        

        #==============================================================================================



		input_file=$(echo "$INPUT" | sed "s/\.in/\_$strain\.in/g")
		output_file=`echo $input_file | sed 's/\.in/\.out/'` 


		if (( $(echo "$strain" == "0" | bc -l) ))
	        then
			previous_input=$INPUT
		else
			previous_input=$(echo "$INPUT" | sed "s/\.in/\_$previous_strain\.in/g")
		fi
	

#=================================================================================================================
#=================================================================================================================
		########### This is where input file manipulations are put #######################################
		########### Allign the left edge of the code with the 3 rows of hashtags##########################
		##################################################################################################

                nat=`grep "nat" $previous_input |awk '{print $3}'`
                positions0=$(grep "ATOMIC_POSITIONS" -A${nat} $INPUT | sed '1d')
                positions=$(grep "ATOMIC_POSITIONS" -A${nat} $previous_input | sed '1d')


                Si_col=$(echo "$positions" | awk '{print $1}')
                x_col0=$(echo "$positions" | awk '{print $2}')
                y_col0=$(echo "$positions" | awk '{print $3}')
                z_col0=$(echo "$positionsi0" | awk '{print $4}')
                f_col=$(echo "$positions" | awk '{print "1 1 0"}')

                zpos_sum=$(echo "$z_col0" |  paste -s -d "+"| bc -l)
                z_avg=$(echo "$zpos_sum/$nat" | bc -l)

                posdiff=$(echo "$z_col0" | awk -v z="$z_avg" '{printf "%06.11f\n",$1-z}')
                delta_zpos=$(echo "$posdiff" | awk -v p="$strain" '{printf "%06.11f\n",$1*p}')

                modzpos=$(paste <(echo "$z_col0") <(echo "$delta_zpos")|  awk '{printf "%06.9f\n",$1+$2}')

                modpos=$(paste <(echo "$Si_col") <(echo "$x_col0") <(echo "$y_col0") <(echo "$modzpos") <(echo "$f_col"))

                sed -e "/ATOMIC_POSITIONS/,+${nat}d" $previous_input | sed -e '/^ Si  28.086  Si.*.UPF/aATOMIC_POSITIONS angstrom' |awk -v pos="$(echo "$modpos")" '{print}/ATOMIC_POSITIONS/{print pos}' > $input_file

		
		##################################################################################################
                ##################################################################################################
                ##################################################################################################
#=================================================================================================================	
#=================================================================================================================


                out_check=$(ls | grep $output_file)

                if [ -z "$out_check" ];
                then
                        mpirun -np $NPROC $PW -npool $NPOOL -input $input_file >& $output_file

                else
                        job_check=$(grep "JOB DONE" $output_file)

                        if [ -z "$job_check" ];
                        then

                                last_positions=$(grep "ATOMIC_POSITIONS" -A$nat $output_file | sed "s/$output_file//g"| grep "Si"| tail -$nat)
                                nlines_l=$(echo "$last_positions" | wc -l)

                                if (( $(echo "$nlines_l" == "$nat" | bc -l) ))
                                then
                                        sed -e "/ATOMIC_POSITIONS/,+${nat}d" $input_file | sed -e '/^ Si  28.086  Si.*.UPF/aATOMIC_POSITIONS angstrom' |awk -v pos="$last_positions" '{print}/ATOMIC_POSITIONS/{print pos}'  > tmp && mv tmp $input_file
                                fi
                                mpirun -np $NPROC $PW -npool $NPOOL -input $input_file >& $output_file

			else
				positions=$(grep "End final" $output_file -B$natp | sed -e '$ d' | sed -e '1d')
                		sed -e "/ATOMIC_POSITIONS/,+${nat}d" $input_file | sed -e '/^ Si  28.086  Si.*.UPF/aATOMIC_POSITIONS angstrom' |awk -v pos="$positions" '{print}/ATOMIC_POSITIONS/{print pos}'  > tmp && mv tmp $input_file

                        fi
                fi

	done
done
