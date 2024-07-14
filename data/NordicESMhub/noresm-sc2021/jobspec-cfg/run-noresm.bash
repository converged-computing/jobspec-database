#!/usr/bin/env bash
#
# Authors: Anne Fouilloux and Jean Iaquinta
# Copyright University of Oslo 2021
# 
# Example of usage:
# ./run-noresm.bash -m betzy -r f19_f19_mg17 -c NF2000climo -p nn1000k -i /cluster/work/users/jeani/archive
#
usage()
{
    echo "usage: run-noresm.bash -m|--machine betzy -c|--compset NF2000climo"
    echo "                       -r|--res f19_f19_mg17 -p|--project nn1000k"
    echo "                       --name case_name_prefix -i|--prefix /path/to/dir"
    echo "                       [-n|--number-ensembles 10]"
    echo "                       -d|--dry -h|--help"

}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m | --machine )          shift
                                  TARGET_MACHINE="$1"
                                  ;;
        -r | --res )              shift
                                  RES="$1"
                                  ;;
        -p | --project )          shift
                                  PROJECT="$1"
                                  ;;
        -i | --prefix )           shift
                                  PREFIX="$1"
                                  ;;
        -c | --compset )          shift
                                  COMPSET="$1"
                                  ;;
        -n | --number-ensembles ) shift
                                  NENSEMBLES="$1"
                                  ;;
        --name )                  shift
                                  CASE_PREFIX="$1"
                                  ;;
        -d | --dry )              DRY_RUN=echo
                                  ;;
        -h | --help )             usage
                                  exit
                                  ;;
        * )                       usage
                                  exit 1
    esac
    shift
done


if [[ "$TARGET_MACHINE" != '' ]] && [[ "$COMPSET" != '' ]] && [[ "$RES" != '' ]] && [[ "$PROJECT" != '' ]] && [[ "$PREFIX" != '' ]] ; then

    $DRY_RUN rm -rf $PREFIX/release-noresm2.0.2

    $DRY_RUN wget https://github.com/NorESMhub/NorESM/archive/refs/tags/release-noresm2.0.2.tar.gz

    $DRY_RUN tar zxvf release-noresm2.0.2.tar.gz  --directory $PREFIX

    $DRY_RUN mv $PREFIX/NorESM-release-noresm2.0.2 $PREFIX/release-noresm2.0.2
    $DRY_RUN cd $PREFIX/release-noresm2.0.2
    $DRY_RUN rm -rf manage_externals
    $DRY_RUN git clone -b manic-v1.1.8 https://github.com/ESMCI/manage_externals.git
    $DRY_RUN sed -i.bak "s/\'checkout\'/\'checkout\', \'--trust-server-cert\', \'--non-interactive\'/" ./manage_externals/manic/repository_svn.py
    $DRY_RUN ./manage_externals/checkout_externals -v

    echo "Number of ensembles: $NENSEMBLES"

    for node in {1..8}; do
        member=1
        while [ $member -le $NENSEMBLES ]; do

            echo "Node $node and Member $member"
            export JOB_NUM_NODES=$node
            echo "======================================================"
            echo "Target machine: $TARGET_MACHINE"
            echo "Compset: $COMPSET Resolution: $RES NODES: $JOB_NUM_NODES"
            echo "======================================================"
            export CASENAME="$CASE_PREFIX-"$JOB_NUM_NODES'x128p-'$COMPSET'-'$RES'-'$member
            $DRY_RUN cd $PREFIX/release-noresm2.0.2/cime/scripts

            $DRY_RUN ./create_newcase --case $PREFIX/cases/$CASENAME --compset $COMPSET \
	             --res $RES --machine $TARGET_MACHINE --run-unsupported \
		     --handle-preexisting-dirs r --project $PROJECT

            $DRY_RUN cd $PREFIX/cases/$CASENAME

            NUMNODES=-$JOB_NUM_NODES
            $DRY_RUN ./xmlchange --file env_mach_pes.xml --id NTASKS --val ${NUMNODES}
            $DRY_RUN ./xmlchange --file env_mach_pes.xml --id NTASKS_ESP --val 1
            $DRY_RUN ./xmlchange --file env_mach_pes.xml --id ROOTPE --val 0
            $DRY_RUN ./xmlchange STOP_N=1
            $DRY_RUN ./xmlchange STOP_OPTION=nmonths

            $DRY_RUN ./case.setup
            $DRY_RUN ./case.build --skip-provenance-check
            $DRY_RUN ./case.submit
            member=$(($member+1))
        done
    done
else
	usage
fi
