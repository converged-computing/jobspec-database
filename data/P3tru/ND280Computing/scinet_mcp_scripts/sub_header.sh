#!/bin/bash 
#
#############################################################################
#
# Title: SciNet header for job submission and GRID scripts
#
# Usage: This script can be included in other scripts by '. sub_header.sh'
#        It sets all the required variables based on the current location
#        of the calling script and provides functions for runND280 .cfg 
#        file and job script generation, based on the specifications here:
#        # (http://www.t2k.org/nd280/datacomp/mcproductionruns/production004)
#        Specifics of the job script should be defined in the calling scripts 
#        (sub_setup.sh, sub_numc.sh and sub_nd280.sh).
#
#        These functions can also be used with GRID scripts for getting and 
#        getting path and file name information.
#   
#        You must set the ND280 version numbers below, as well as other 
#        flags necessary before production. The arrays specifying directory
#        names should also be specified according to:
#        http://www.t2k.org/nd280/datacomp/mcproductionruns/production004
#        where allowed combinations are generated using the mk_dir.sh 
#        script.
#
#
#############################################################################

# ND280 version numbers and root location
# (Make sure ${ND280SOFT_DIR}/setup_t2knd280.sh exists for setting up ND280 environment)
ND280VERS=v9r7p9
ND280CONTROLVERS=v1r47
#ND280SOFT_DIR=${SCRATCH}/nd280_${ND280VERS}
ND280SOFT_DIR=/location/of/nd280/installation

# Set this to 0 when testing to check the .cfg files and job scripts produced
SUBJOB_FOR_REAL=0
# VERIFY=1 implies we are generating test files to be placed 
# in the verify/nd280vers directory
VERIFY=0

# Flag to output and copy core dumps 
DEBUG=0

# Starting counter label for submitted jobs
BATCHIN=0

# Flag to resubmit missing and/or failed jobs
SUBMISS=0

# Files I want to intentionally skip
SKIPFILE=${HERE}/fileskip.txt


# Arrays specifiying directories to create
# (Placed in header to be synced with mk_dir.sh and sub_setup.sh)
RESPINS=(C)
SIMNAMES=(genie neut)
BASELINES=(2010-02-water 2010-11-water 2010-11-air)
VOLUMES=(magnet basket)
MC_TYPES=(beama beamb beam nue ncpizero ccpizero ncpiplus ccpiplus)
STEPS=(nd280)


# Input 'numc' production directory including <MC name>/<respin>/verify/nd280vers
# For nd280, this is used for finding the 'numc' vectors.
# For numc generation, it is used for finding the generator setup files.
#export INPUT_NUMC_DIR=/scratch/j/jmartin/pdeperio/nd280/production004/A/mcp
export INPUT_NUMC_DIR=/location/of/input/numc/files

# NEUT Version number
NEUT_VERS=5.1.4.1

# Set the path of the NEUT installation
# (Make sure ${NEUT_ROOT}/neutsmpl/setup.sh and /neutgeom/setup.sh exists which setups up the NEUT running environment properly)
#export NEUT_ROOT="/home/j/jmartin/pdeperio/lib/neut/neut_${NEUT_VERS}/src"
export NEUT_ROOT=/location/of/neut/installation/src



# GENIE Version number
GENIE_VERS=2.6.4

# Set the path of the GENIE installation
# (Make sure ${GENIE}/setup.sh exists to setup up running environment)
#export GENIE="/home/j/jmartin/pdeperio/lib/genie_${GENIE_VERS}"
export GENIE=/location/of/genie/installation

# Set path of GENIE xsec table (pre-calculated from free nucleon 
# splines via genie_setup_xsec_tables.cfg)
#export GENIE_XS_TABLE="/scratch/j/jmartin/pdeperio/nd280/production004/genie_setup/genie_nd280_all_xs_table_${GENIE_VERS}.xml"
export GENIE_XS_TABLE=/location/of/genie/splines.xml




# Flux file information, directory and filenaming conventions
FLUX_ROOT_PATH=/scratch/j/jmartin/pdeperio/flux_11a

# This assumes the flux files have been hadd'ed already
FLUX_DETIDS=(5 6)
FLUX_NFILES=(500 250)
FLUX_FILENAMES=(nu.nd${FLUX_DETIDS[0]}_flukain.0-$[${FLUX_NFILES[0]}-1] nu.nd${FLUX_DETIDS[1]}_flukain.0-$[${FLUX_NFILES[1]}-1])

FLUX_BASKET_PATH=${FLUX_ROOT_PATH}/nd${FLUX_DETIDS[0]}/hadd/${FLUX_FILENAMES[0]}.root
FLUX_MAGNET_PATH=${FLUX_ROOT_PATH}/nd${FLUX_DETIDS[1]}/hadd/${FLUX_FILENAMES[1]}.root
FLUX_FULL_PATH=(${FLUX_BASKET_PATH} ${FLUX_MAGNET_PATH})
#FLUX_INGRID_PATH=${FLUX_ROOT_PATH}/nd34


# Location of nd280Control setup.sh file (Make sure to set ND280(Control) 
# version numbers in the calling script
RUNND280_SETUP=${ND280SOFT_DIR}/nd280Control/${ND280CONTROLVERS}/cmt/setup.sh



# The RAM disk on each worker node
# You can use this to read and write generator setup and 'numc' files since 
# NEUT/GENIE memory requirement is low enough. Do not use for nd280 chain, or 
# else the jobs will run out of memory.
export SCRDIR=/dev/shm


# Current directory
HERE=`pwd`


# Set the input 'numc' base directory
# WARNING: Be careful when using 'numc' files from different productions
function set_numc_dir {

    # All samples now have their dedicated 'numc' samples
    NUMC_DIR=${INPUT_NUMC_DIR}/${SIMNAME}/${BASELINE}/${vol}/${MC_TYPE}/numc

}


# Function to set beam spill and bunch parameters for ND280mc.
# Currently has meaning for only the magnet sample (see function
# get_mc_details() where mc_full_spill is set).
# Currently calculated using NEUT v5.1.0, GENIE v2.6.2 and 
# ND280 v9r7p1.
function set_beam {

    # Interactions per spill number (Int/Sp) comes from the expected 
    # event rate (Evts/POT) after running each generators' setup program 
    # (e.g. with sub_setup.sh) on the entire flux file set
    # given the desired beam power (P):
    #
    #    Int/Sp = (Evts/POT) * (POT/Sp)
    #    POT/Sp = P * T / E
    # 
    #       where beam power, P = [50, 100,..] kW
    #             Rep. Period,  T = [3.52, 3.2] seconds
    #             proton energy, E = 30 GeV
    #
    # Note: This should really be implemented into an external
    #       bunch building program (oaBunchBuilder)
    #
    # The beam spec values are summarized at MC production page:
    # http://www.t2k.org/nd280/datacomp/mcproductionruns/production004
    #

    # Check that beam spec was defined properly in path
    if [ "${BEAM_SPEC}" == "" ]; then
	echo "Beam specification not defined in path"
	exit 1
    fi

    # ND280 Run 1 (Beam Spec 'a': P = 50 kW, R = 3.52 seconds)
    if [[ $ND280_RUN == 1 && $BEAM_SPEC == a ]]; then

	NBUNCHES=6
	POT_PER_SPILL=3.66169e13
	
	if [ $P0D_NAME == water ]; then
	    
	    if [ "${SIMNAME}" == "neut" ]; then
		INTERACTIONS_PER_SPILL=4.01207

	    elif [ "${SIMNAME}" == "genie" ]; then
		INTERACTIONS_PER_SPILL=3.62166

	    fi		    
	    
	elif [ $P0D_NAME == air ]; then
	    
	    if [ "${SIMNAME}" == "neut" ]; then
		INTERACTIONS_PER_SPILL=4.00068

	     # Scaled by NEUT air/water ratio (should not be needed for MCP4)
	    elif [ "${SIMNAME}" == "genie" ]; then
		INTERACTIONS_PER_SPILL=3.61138

	    fi		    
	    
	fi


        # Duration/width of each bunch in ns
        BUNCH_DURATION=17

	
    # ND280 Run 2 (Beam Spec 'b': P = 100 kW, R = 3.2 seconds)
    elif [[ $ND280_RUN == 2 && $BEAM_SPEC == b ]]; then
	
	NBUNCHES=8
	POT_PER_SPILL=6.65761e13

	if [ $P0D_NAME == water ]; then
	    
	    if [ "${SIMNAME}" == "neut" ]; then
		INTERACTIONS_PER_SPILL=7.71184

            # GENIE needs to be updated
	    elif [ "${SIMNAME}" == "genie" ]; then
		INTERACTIONS_PER_SPILL=6.95933

	    fi		    
	    
	elif [ $P0D_NAME == air ]; then
	    
	    if [ "${SIMNAME}" == "neut" ]; then
		INTERACTIONS_PER_SPILL=7.69114

            # GENIE needs to be updated
	    elif [ "${SIMNAME}" == "genie" ]; then
		INTERACTIONS_PER_SPILL=6.94061

	    fi		    
	    
	fi
	
        # Duration/width of each bunch in ns
        BUNCH_DURATION=19

    # Error
    else
	echo "Unknown ND280 Run (${ND280_RUN}) + Beam Spec combination (${BEAM_SPEC}) "
	exit
    fi

    
    
    
    # Offset from beam trigger in ns
    TIME_OFFSET=50
    
    
    # Following are currently not changed from default and thus are not included 
    # in the original .cfg files

    # Time separation of bunches in ns
    BUNCH_SEPARATION=582
    
}


# Function to set the run number conventions for various samples
function set_run_numbers {

    # RUNSTART and RUNEND should be specified by command line argument
    # and padded with zeros for 7 digits, and the 8th digit is 9
    # e.g. RUNSTART=9`printf "%07d" $1`
    # (See the calling script you are using for syntax)
    if [[ "$RUNSTART" == "" || "$RUNEND" == "" ]]; then
	echo "Must specify RUNSTART and/or RUNEND"
	exit 1
    fi


    if [[ "$SUBRUNSTART" == "" || "$SUBRUNEND" == "" ]]; then
	echo "Must specify SUBRUNSTART and/or SUBRUNEND"
	exit 1
    fi


    # GENIE
    if [ $SIMNAME == "genie" ]; then
	RUNSTART=$(( $RUNSTART + 1000000 ))
	RUNEND=$(( $RUNEND + 1000000 ))
    fi

    # ND280 Run
    RUNSTART=$(( ${ND280_RUN}*100000 + ${RUNSTART} ))
    RUNEND=$(( ${ND280_RUN}*100000 + ${RUNEND} ))

    # P0D water/air
    RUNSTART=$(( ${P0D_WATER}*10000 + ${RUNSTART} ))
    RUNEND=$(( ${P0D_WATER}*10000 + ${RUNEND} ))


    # Add to run index for special basket samples
    ADD_RUN=0
    if [ "${vol}" == "basket" ]; then
	if [ "${MC_TYPE}" == "beam" ]; then
	    ADD_RUN=1000
	elif [ "${MC_TYPE}" == "nue" ]; then
	    ADD_RUN=2000
	elif [ "${MC_TYPE}" == "ncpizero" ]; then
	    ADD_RUN=3000
	elif [ "${MC_TYPE}" == "ccpizero" ]; then
	    ADD_RUN=4000
	elif [ "${MC_TYPE}" == "ncpiplus" ]; then
	    ADD_RUN=5000
	elif [ "${MC_TYPE}" == "ccpiplus" ]; then
	    ADD_RUN=6000
	else
	    echo "Unknown MC type for basket sample: " ${MC_TYPE}
	    exit 1
	fi

	RUNSTART=$(( $RUNSTART + $ADD_RUN ))
	RUNEND=$(( $RUNEND + $ADD_RUN ))
    fi


    # Variables controlling how many runs to increment in a given 
    # script call.
    #
    # RUN_INCR not needed for Production004 and beyond, since we 
    # are combining all flux files, so it is set to 1.
    #
    # nIntFileCombine is the number of interaction files to combine 
    # for a given ND280MC run (for increasing the ND280 output file size 
    # from small, cherry picked, interaction files). For now, we re-generate
    # 'numc' files, so the statistics is increased there rather 
    # than combining files (nIntFileCombine=1)
    if [ ${STEP} == "numc" ]; then
	RUN_INCR=1
	nIntFileCombine=1
    else

        # Beam and nue
	if [[ ${MC_TYPE} == beam* || ${MC_TYPE} == nu* ]]; then
	    RUN_INCR=1
	    nIntFileCombine=1

        # Cherry picked
	else
	    RUN_INCR=1
	    nIntFileCombine=1
	fi
    fi
    
    echo "Run range: " ${RUNSTART} "-" ${RUNEND}
    echo "Subrun range": $SUBRUNSTART "-" $SUBRUNEND
    echo ""
}

function get_data_run {
    run_subrun=`echo ${datafile} | cut -d'_' -f4`
    run=`echo ${run_subrun} | cut -d'-' -f1`
    subrun_label=`echo ${run_subrun} | cut -d'-' -f2`
    if [ ${subrun_label} == "0000" ]; then
	subrun=0
    else
	subrun="$(echo $subrun_label | sed 's/0*//')"
    fi
}

function grid_run_skip {
    if [[ $run -lt $RUNSTART || $run -gt $RUNEND ]]; then
	continue
    fi
    
    if [[ $run -eq $RUNSTART && $subrun -lt $SUBRUNSTART ]]; then
	continue
    fi
    
    if [[ $run -eq $RUNEND && $subrun -gt $SUBRUNEND ]]; then
	continue
    fi
}

capitalize_ichar ()          #  Capitalizes initial character
{                            #+ of argument string(s) passed.

  string0="$@"               # Accepts multiple arguments.

  firstchar=${string0:0:1}   # First character.
  string1=${string0:1}       # Rest of string(s).

  FirstChar=`echo "$firstchar" | tr a-z A-Z`
                             # Capitalize first character.

  echo "$FirstChar$string1"  # Output to stdout.

} 


function get_mc_details {

    # Get directory level of the production### directory
    for (( prod_dir_level=1; prod_dir_level<99; prod_dir_level++ ))
      do

	this_level=`echo ${PRODUCTION_DIR} | cut -d'/' -f${prod_dir_level}`
	if [[ "${this_level}" == production* ]]; then
	    break;
	fi
    done
    
    MC_NAME=`echo ${HERE} | cut -d'/' -f${prod_dir_level}`
    RESPIN=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+1]`
    MCP=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+2]`

    # Extra two levels for verification structure except for 'numc'
    if [ $VERIFY -eq 1 ]; then

	VERIFY_NAME=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+3]`
	if [ "${VERIFY_NAME}" != verify ]; then
	    echo "Is your \"verify\" directory named properly=(${VERIFY_NAME})?"
	    exit 1
	fi

	ND280VERS_CHECK=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+4]`
	if [ "${ND280VERS_CHECK}" != ${ND280VERS} ]; then
	    echo "Directory ND280 version (${ND280VERS_CHECK}) does not match specified version (${ND280VERS})."
	    exit 1
	fi
	
	let prod_dir_level=${prod_dir_level}+2
	
    fi

    SIMNAME=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+3]`
    BASELINE=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+4]`
    vol=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+5]`
    MC_TYPE=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+6]`
    STEP=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+7]`
    SETUP=`echo ${HERE} | cut -d'/' -f$[${prod_dir_level}+8]`

    # Capitalize volume for CFG files
    Vol_up=`capitalize_ichar "${vol}"`
    
    # For numc setup and generation
    FLUX_REGION=${vol}
    # Select array index for flux names
    if [ "$FLUX_REGION" == "magnet" ]
	then
	idetid=1
    elif [ "$FLUX_REGION" == "basket" ]
	then
	idetid=0
    else
	echo "Invalid Flux Region (vol) specified: $FLUX_REGION"
	exit
    fi

    # Neutrino type for numc generation
    if [[ ${MC_TYPE} == nu* ]]; then
	NU_TYPE=${MC_TYPE}

    # Full beam or Files to be cherry picked (pion only)
    elif [[ ${MC_TYPE} == beam* || ${MC_TYPE} == *pi* ]]; then
	NU_TYPE=beam

    else
	echo "Unknown MC_TYPE = ${MC_TYPE}"
	exit 1
    fi

    # Geometry baseline
    BASELINE_CFG=`echo "${BASELINE}" | cut -d'-' -f1`
    BASELINE_NAME=${BASELINE_CFG}

    # If baseline is some year-month format (YYYY-MM)
    if [ $BASELINE_CFG != "full" ]; then
	YEAR=$BASELINE_CFG
	MONTH=`echo "${BASELINE}" | cut -d'-' -f2`

	BASELINE_CFG="${YEAR}-${MONTH}"
	BASELINE_NAME="${YEAR}${MONTH}"

    # Do not allow "full" baseline, must specify running period
    # in order for beam spill and bunch information to be set 
    # properly
    else
	echo "Error: Must specify year-month for baseline"
	exit
    fi


    # Define the ND280 Run for Production 004.
    # This needs to be modified to include future runs 
    # and changes in run definitions.
    if [ $YEAR == 2010 ]; then
	if [ $MONTH == 02 ]; then
	    
	    ND280_RUN=1
	    
	elif [ $MONTH == 11 ]; then
	    
	    ND280_RUN=2

	# Error in month definition
	else
	    echo "Unknown month=${MONTH} in year ${YEAR} baseline definition"
	    exit 1
	fi

    # Error in year definition
    else
	echo "Unknown year=${YEAR} in baseline defintion"
	exit 1
    fi




    # P0D Configuration
    if [[ ${BASELINE} == *water ]]; then
	P0D_WATER=1
	P0D_NAME="water"
    elif [[ ${BASELINE} == *air ]]; then
	P0D_WATER=0
	P0D_NAME="air"
    else
	echo "Unknown P0D configuration: " ${BASELINE}
	exit 1
    fi

    # Define generator type
    if [ $SIMNAME == "neut" ]; then
	GEN_TYPE=Neut_RooTracker
    elif [ $SIMNAME == "genie" ]; then
	GEN_TYPE=Genie
    else
	echo "Unknown neutrino generator:" $SIMNAME
	exit 1
    fi

    
    # Beam specification
    if [[ ${MC_TYPE} == beam* ]]; then
	BEAM_SPEC=${MC_TYPE:4}
    fi
    

    # ND280mc beam settings
    if [ ${STEP} == "nd280" ]; then
	if [ "${vol}" == "basket" ]; then

	    if [ "${BEAM_SPEC}" != "" ]; then
		echo "Error: There should be no beam spec in basket samples"
		exit 1
	    fi

	    MC_FULL_SPILL=0
	    COUNT_TYPE=FIXED
	    INTERACTIONS_PER_SPILL=1
	    POT_PER_SPILL=0
            #set_beam
            NBUNCHES=1
            TIME_OFFSET=0
            BUNCH_DURATION=0

	elif [ "${vol}" == "magnet" ]; then
	    MC_FULL_SPILL=1
	    COUNT_TYPE=MEAN
	    set_beam

	else
	    echo "No ND280mc configuration for volume: " ${vol}
	    exit 1
	fi
    fi
    
    # Cherry Picker settings
    if [[ ${MC_TYPE} == beam* || ${MC_TYPE} == nu* ]]; then
	CHERRY_PICKED=0

    else
	CHERRY_PICKED=1
	if [ "${MC_TYPE}" == "ncpizero" ]; then
	    NUM_MESONS=0
	    NUM_LEPTONS=0
	    NUM_MU_MINUS=0
	    NUM_PIZERO=1
	    NUM_PIPLUS=0
	elif [ "${MC_TYPE}" == "ccpizero" ]; then
	    NUM_MESONS=0
	    NUM_LEPTONS=1
	    NUM_MU_MINUS=1
	    NUM_PIZERO=1
	    NUM_PIPLUS=0
	elif [ "${MC_TYPE}" == "ncpiplus" ]; then
	    NUM_MESONS=0
	    NUM_LEPTONS=0
	    NUM_MU_MINUS=0
	    NUM_PIZERO=0
	    NUM_PIPLUS=1
	elif [ "${MC_TYPE}" == "ccpiplus" ]; then
	    NUM_MESONS=0
	    NUM_LEPTONS=1
	    NUM_MU_MINUS=1
	    NUM_PIZERO=0
	    NUM_PIPLUS=1
	else
	    echo "No cherry picker for MC type: " ${MC_TYPE}
	    exit 1
	fi
    fi


    
    echo ""
    echo ${MC_NAME}-${MCP} " SciNet Script"

    if [ $VERIFY -eq 1 ]; then
	echo "Verification ND280 Version ${ND280VERS}"
    fi
    
    echo "Generator name: " ${SIMNAME} ${SETUP}
    echo "Volume: " ${vol}
    echo "Baseline: " ${BASELINE}
    echo "MC type: " ${MC_TYPE} 
    echo "Step: " ${STEP}
    echo ""
}


# Function to set the ND280 software chain module list
function set_modules {
    
    ALL_MODULES=(oaCherryPicker nd280MC elecSim oaCalibMC oaRecon oaAnalysis)
    MODULE_FILE=(nucp g4mc elmc cali reco anal)

    # Default is to run the full chain
    if [[ "${MODULE_START}" == "" && "${MODULE_END}" == "" ]]; then
	
	# No cherry-picking
	if [[ ${CHERRY_PICKED} == 0 ]]; then
	    MODULE_LIST=`echo ${ALL_MODULES[@]:1}`
	    
        # Cherry-picking
	else
	    MODULE_LIST=`echo ${ALL_MODULES[@]}`
	    
	fi

    # Specified start (and end) module
    else
	
	imodule=0
	i_module_start=0
	i_module_end=0
	for module in ${ALL_MODULES[@]}
	  do

	  if [[ "${MODULE_START}" == "${module}" ]]; then
	      i_module_start=${imodule}
	  fi

	  # Run until the end of the chain
	  if [[ "${MODULE_END}" == "" ]]; then
	      i_module_end=5  # Last index of the ALL_MODULES array

	  # Run until specified end module
	  elif [[ "${MODULE_END}" == "${module}" ]]; then
	      i_module_end=${imodule}
	  fi
	  
	  let imodule=${imodule}+1
	done
	
	# Check that specified modules actually exist
	if [[ ${i_module_start} == 0 ]]; then 
	    echo Error: Starting module ${MODULE_START} is undefined.
	    exit
	fi
	if [[ ${i_module_end} == 0 ]]; then 
	    echo Error: Ending module ${MODULE_END} is undefined.
	    exit
	fi

	# Check that modules were specified in the correct order
	if [[ ${i_module_end} < ${i_module_start} ]]; then
	    echo Error: End module ${MODULE_END} should not be called before start module ${MODULE_START}
	    exit
	fi

	
	# Bash substring command syntax expects number of elements following the starting element
	let i_module_count=${i_module_end}-${i_module_start}+1
	MODULE_LIST=`echo ${ALL_MODULES[@]:${i_module_start}:${i_module_count}}`

    fi
    
}


# Function to specify the POT per interaction file
function set_pot {

    if [ ${vol} == "magnet" ]; then
	POT=5e17

    elif [ ${vol} == "basket" ]; then

	if [ ${MC_TYPE} == beam ]; then
	    POT=1e18
	    
	elif [[ ${MC_TYPE} == nue ]]; then
	    POT=2E19

        elif [[ ${MC_TYPE} == ncpizero ]]; then
            POT=2E19

        elif [[ ${MC_TYPE} == ccpizero ]]; then
            POT=2E19

        elif [[ ${MC_TYPE} == ncpiplus ]]; then
            POT=5E19

        elif [[ ${MC_TYPE} == ccpiplus ]]; then
            POT=1E19

	else
	    echo "Unknown MC_TYPE=${MC_TYPE} while setting POT"
	    exit 1
	fi
    else
	echo "set_pot Error: Unknown volume for beam:" ${vol}
	exit
    fi
   
}


# Function to set the comment in the output filenames
function set_comment {

    # Assuming we're in "production###" naming era
    COMMENT=prod${MC_NAME:10:3}${vol}${BASELINE_NAME}${P0D_NAME}    

    # Magnet sample with specified beam power
    if [ "${vol}" == "magnet" ]; then

        COMMENT=${COMMENT}${BEAM_SPEC}

    fi
	    
    # Cherry Picked 
    if [[ ${MC_TYPE} != beam* && ${MC_TYPE} != nu* ]]; then
	
        COMMENT=${COMMENT}${MC_TYPE}

    fi

    # Generator setup comment 
    if [[ "${SETUP}" == "setup" ]]; then
	COMMENT=${COMMENT}${SIMNAME}setup
    fi
    
    # Make it all lower case
    COMMENT=`echo "${COMMENT}" | tr '[:upper:]' '[:lower:]'`
}


# Function to specify the WALLTIME and number of 
# processors per node to use for the current job
function set_walltime {

# For neutrino vector generation
if [ "${STEP}" == "numc" ]; then

    # Setting up the generators
    if [ "${SETUP}" == "setup" ]; then

	WALLTIME=5:00:00
	NPROCS=8	

    # Actual numc vector generation
    else

	# Basket volume
	if [ "${vol}" == "basket" ]; then
	
            if [ ${MC_TYPE} == beam ]; then
		WALLTIME=2:00:00

	    elif [[ ${MC_TYPE} == nue ]]; then
		WALLTIME=4:00:00
		
	    elif [[ ${MC_TYPE} == ncpizero ]]; then
		WALLTIME=8:00:00
		
	    elif [[ ${MC_TYPE} == ccpizero ]]; then
		WALLTIME=8:00:00
		
	    elif [[ ${MC_TYPE} == ncpiplus ]]; then
		WALLTIME=12:00:00
		
	    elif [[ ${MC_TYPE} == ccpiplus ]]; then
		WALLTIME=6:00:00

	    fi
	    
	    NPROCS=8
	    
	 # Magnet volume
	elif [ "${vol}" == "magnet" ]; then
	    
	    WALLTIME=8:00:00
	    NPROCS=8
	 
	# Error
	else
	    echo "No job definition for volume: " ${vol}
	    exit 1
	fi
    fi

# For ND280 event generation
else 

    # oaAnalysis-only Processing
    if [ "${MODULE_LIST[0]}" == "oaAnalysis" ]; then
        WALLTIME=2:00:00
        NPROCS=6
    else   

    # Basket volume
    if [ "${vol}" == "basket" ]; then
	
	# NCpi0 cherry picked
	if [ "${MC_TYPE}" == "ncpizero" ]; then
	    WALLTIME=6:00:00
	    NPROCS=6

	# CCpi0 cherry picked
	elif [ "${MC_TYPE}" == "ccpizero" ]; then
	    WALLTIME=8:00:00
	    NPROCS=6

        # NCpi+ cherry picked
        elif [ "${MC_TYPE}" == "ncpiplus" ]; then
            WALLTIME=6:00:00
            NPROCS=6

        # CCpi+ cherry picked
        elif [ "${MC_TYPE}" == "ccpiplus" ]; then
            WALLTIME=9:00:00
            NPROCS=6


	# Nue from specially generated interaction files
	elif [ "${MC_TYPE}" == "nue" ]; then
	    WALLTIME=8:00:00
	    NPROCS=6

	# Full beam sample
	else
	    WALLTIME=8:00:00
	    NPROCS=7   # This may need to be optimized due to memory limitations
	fi

    # Magnet volume
    elif [ "${vol}" == "magnet" ]; then
	WALLTIME=15:00:00
	NPROCS=6
    
    # Error
    else
	echo "No job definition for volume: " ${vol}
	exit 1
    fi

    fi
fi
}



# Function to set the .cfg filename
function set_cfg_filename {

    # Interaction generation 
    if [[ "${STEP}" == "numc" ]]; then
	CFG_PREFIX=${SIMNAME}
	
	# Generator setup (max interaction probability calc.)
	if [[ "${SETUP}" == "setup" ]]; then
	    CFG_PREFIX=${CFG_PREFIX}_setup
	fi

    # ND280 event generation
    else

	# Beam or nue
	if [[ ${CHERRY_PICKED} == 0 ]]; then
	    CFG_PREFIX=nd280
	    
	# Cherry picked
	else
	    CFG_PREFIX=nd280_cp
	fi
    fi

    # Generator setup .cfg file
    if [[ "${SETUP}" == "setup" ]]; then
	CFG_FILE=${CFG_PREFIX}_vol${vol}_${BASELINE}.cfg

    # Everything else
    else
	CFG_FILE=${CFG_PREFIX}_${MC_TYPE}_${vol}_${BASELINE}_${run}-${subrun}.cfg
	
	# For subrun looping within a job script
	CFG_FILE_SUB=${CFG_PREFIX}_${MC_TYPE}_${vol}_${BASELINE}_${run}-\${subrun}.cfg
    fi    
    
}




# Function to make .cfg files
function mk_cfg {

    # Make a new copy of the original .cfg file 
    cp ${CFG_FILE_ORIG} ${CFG_FILE_PATH}

# [software]
    sed -i "s|cmtpath = cmtpath|cmtpath = ${ND280SOFT_DIR}|g" ${CFG_FILE_PATH}
    sed -i "s|cmtroot = cmtroot|cmtroot = ${ND280SOFT_DIR}/CMT/v1r20p20081118|g" ${CFG_FILE_PATH}
    sed -i "s|nd280ver = nd280ver|nd280ver = ${ND280VERS}|g" ${CFG_FILE_PATH}
    sed -i "s|genie_setup_script = genie_setup_script|genie_setup_script = ${GENIE}/setup.sh|g" ${CFG_FILE_PATH}
    
    # NEUT
    sed -i "s|neut_setup_script = neut_setup_script|neut_setup_script = ${NEUT_ROOT}/neutgeom/setup.sh|g" ${CFG_FILE_PATH}

    # GENIE
    sed -i "s|genie_setup_script = genie_setup_script|genie_setup_script = ${GENIE}/setup.sh|g" ${CFG_FILE_PATH}

# [configuration]

    # Module list defined in set_modules()
    sed -i "s/module_list = module_list/module_list = ${MODULE_LIST}/g" ${CFG_FILE_PATH}

    # This is assuming a single input file, defined in the calling script
    sed -i "s|inputfile = inputfile|inputfile = ${INPUTFILE[0]}|g" ${CFG_FILE_PATH}

# [filenaming]
    sed -i "s/run_number = 0/run_number = ${run}/g" ${CFG_FILE_PATH}
    sed -i "s/subrun = 0/subrun = ${subrun}/g" ${CFG_FILE_PATH}
    sed -i "s/comment = comment/comment = ${COMMENT}/g" ${CFG_FILE_PATH}

# [geometry]
    sed -i "s/baseline = baseline/baseline = ${BASELINE_CFG}/g" ${CFG_FILE_PATH}
    sed -i "s/p0d_water_fill = p0d_water_fill/p0d_water_fill = ${P0D_WATER}/g" ${CFG_FILE_PATH}

# [neutrino]
    if [ "${STEP}" == "numc" ]; then
	
	# Check if flux file exists
	if [ ! -s ${FLUX_FULL_PATH[$idetid]} ]; then
	    echo "Flux file does not exist: " ${FLUX_FULL_PATH[$idetid]} 
	    exit
	fi
	sed -i "s|flux_file = flux_file|flux_file = ${FLUX_FILENAMES[$idetid]}.root|g" ${CFG_FILE_PATH}

	RND=`echo "scale=9;$RANDOM/32767*900000000" | bc | cut -d'.' -f1` 
	sed -i "s/random_seed = nu_seed/random_seed = ${RND}/g" ${CFG_FILE_PATH}
	sed -i "s/neutrino_type = beam/neutrino_type = ${NU_TYPE}/g" ${CFG_FILE_PATH}
	sed -i "s/master_volume = Magnet/master_volume = ${Vol_up}/g" ${CFG_FILE_PATH}
	sed -i "s/flux_region = Magnet/flux_region = ${FLUX_REGION}/g" ${CFG_FILE_PATH}
	sed -i "s/pot = pot/pot = ${POT}/g" ${CFG_FILE_PATH}

	# Name of generator setup files
	MAXINT_FILE=${FLUX_FILENAMES[$idetid]}_${SIMNAME}_evtrate_${BASELINE}.root
	MAXINT_FILE=`echo "${MAXINT_FILE}" | tr '[:upper:]' '[:lower:]'`

	# Check if setup file exists
	if [ ! -s ${NUMC_DIR}/setup/${MAXINT_FILE} -a "${SETUP}" == "" ]; then
	    echo "${SIMNAME} setup file does not exist: ${NUMC_DIR}/setup/${MAXINT_FILE}" 
	    exit
	fi	
	    
	# NEUT specific commands
	if [ ${SIMNAME} == "neut" ]; then

	    # Interaction probability setup file
	    sed -i "s|maxint_file = maxint_file|maxint_file = ${MAXINT_FILE}|g" ${CFG_FILE_PATH}
 
	    RND=`echo "scale=9;$RANDOM/32767*900000000" | bc | cut -d'.' -f1` 
	    sed -i "s/neut_seed1 = 0/neut_seed1 = ${RND}/g" ${CFG_FILE_PATH}

	    RND=`echo "scale=9;$RANDOM/32767*900000000" | bc | cut -d'.' -f1` 
	    sed -i "s/neut_seed2 = 0/neut_seed2 = ${RND}/g" ${CFG_FILE_PATH}

	    RND=`echo "scale=9;$RANDOM/32767*900000000" | bc | cut -d'.' -f1` 
	    sed -i "s/neut_seed3 = 0/neut_seed3 = ${RND}/g" ${CFG_FILE_PATH}

	# GENIE specific commands
	elif [ ${SIMNAME} == "genie" ]; then

	    # POT of input flux file needs to be specified for GENIE (NEUT assumes 1E21 POT 
	    # for single flux files, and accounts for hadd'ed flux files)
	    FLUX_FILE_POT=${FLUX_NFILES[$idetid]}E21
	    sed -i "s/flux_file_pot = flux_file_pot/flux_file_pot = ${FLUX_FILE_POT}/g" ${CFG_FILE_PATH}

	    # Xsec table filename hardcoded at the top of this file
	    if [ ! -e ${GENIE_XS_TABLE} ]; then
		echo "GENIE xsec table does not exist: ${GENIE_XS_TABLE}"
		exit
	    fi
	    sed -i "s|genie_xs_table = genie_xs_table|genie_xs_table = ${GENIE_XS_TABLE}|g" ${CFG_FILE_PATH}

	    # Interaction probability setup file
	    sed -i "s|genie_flux_probs_file_name = genie_flux_probs_file_name|genie_flux_probs_file_name = ${MAXINT_FILE}|g" ${CFG_FILE_PATH}

	    
	    # Deprecated (use flux interaction probability pre-calculation)
	    #PATH_FILE=genie_nd280_${vol}_paths_${GENIE_VERS}.xml
	    #sed -i "s|genie_paths = genie_paths|genie_paths = ${GENIE_SETUP_DIR}/${PATH_FILE}|g" ${CFG_FILE_PATH}

	    
	    
	else
	    echo "Error: Unknown SIMNAME: " ${SIMNAME}
	    exit
	fi

    fi

# [nd280mc]
    RND=`echo "scale=9;$RANDOM/32767*900000000" | bc | cut -d'.' -f1` 
    sed -i "s/random_seed = mc_seed/random_seed = ${RND}/g" ${CFG_FILE_PATH}
    sed -i "s/mc_type = mc_type/mc_type = ${GEN_TYPE}/g" ${CFG_FILE_PATH}
    sed -i "s/mc_full_spill = mc_full_spill/mc_full_spill = ${MC_FULL_SPILL}/g" ${CFG_FILE_PATH}
    sed -i "s/nbunches = nbunches/nbunches = ${NBUNCHES}/g" ${CFG_FILE_PATH}
    sed -i "s/interactions_per_spill = interactions_per_spill/interactions_per_spill = ${INTERACTIONS_PER_SPILL}/g" ${CFG_FILE_PATH}
    sed -i "s/pot_per_spill = pot_per_spill/pot_per_spill = ${POT_PER_SPILL}/g" ${CFG_FILE_PATH}
    sed -i "s/time_offset = time_offset/time_offset = ${TIME_OFFSET}/g" ${CFG_FILE_PATH}
    sed -i "s/bunch_duration = bunch_duration/bunch_duration = ${BUNCH_DURATION}/g" ${CFG_FILE_PATH}
    sed -i "s/count_type = count_type/count_type = ${COUNT_TYPE}/g" ${CFG_FILE_PATH}

# [electronics]
    RND=`echo "scale=9;$RANDOM/32767*900000000" | bc | cut -d'.' -f1`
    sed -i "s/random_seed = elec_seed/random_seed = ${RND}/g" ${CFG_FILE_PATH}
    
# [analysis]
    sed -i "s|pass_through_dir = pass_through_dir|pass_through_dir = ${PASS_THROUGH_DIR}|g" ${CFG_FILE_PATH}
     
# [cherry_picker]
    sed -i "s|num_mesons = num_mesons|num_mesons = ${NUM_MESONS}|g" ${CFG_FILE_PATH}
    sed -i "s|num_leptons = num_leptons|num_leptons = ${NUM_LEPTONS}|g" ${CFG_FILE_PATH}
    sed -i "s|num_mu_minus = num_mu_minus|num_mu_minus = ${NUM_MU_MINUS}|g" ${CFG_FILE_PATH}
    sed -i "s|num_pizero = num_pizero|num_pizero = ${NUM_PIZERO}|g" ${CFG_FILE_PATH}
    sed -i "s|num_piplus = num_piplus|num_piplus = ${NUM_PIPLUS}|g" ${CFG_FILE_PATH}

    # This is filled in sub_nd280.sh
    INPUTFILE_LIST=`echo ${INPUTFILE[@]}`
    sed -i "s|inputfile_list = inputfile_list|inputfile_list = ${INPUTFILE_LIST}|g" ${CFG_FILE_PATH}

# Done

    echo "Created CFG file: ${CFG_FILE_PATH}"
}




function mk_subfile_header {

    # Header should only be created once per SUBFILE
    if [ $PROCS -eq 0 ]; then
    
	# Name that shows up with 'qstat'
	if [ "${SETUP}" == "setup" ]; then
	    JOBNAME=${CFG_PREFIX}${COMMENT}.setup.${BATCH}
	else
	    #JOBNAME=${CFG_PREFIX}${COMMENT}${run}.${BATCH}
            JOBNAME=${COMMENT}${run}.${BATCH}
	fi

# Following applies to all types of jobs and specifies
# the walltime and sources the ND280 software setup scripts
cat > ${SUBFILE}<<EOF
#!/bin/bash

#PBS -l nodes=1:ppn=8,walltime=${WALLTIME}
#PBS -N ${JOBNAME}

source ${ND280SOFT_DIR}/setup_t2knd280.sh
source ${RUNND280_SETUP}

cd ${WORKDIR}

EOF

        # Generate core dumps
        if [ ${DEBUG} -eq 1 ]; then
cat >> ${SUBFILE}<<EOF
ulimit -c unlimited

EOF
        fi

        # Additional header commands for interaction generator 'numc' jobs
        if [ "${STEP}" == "numc" ]; then

            # Setup script for setting g77 compiler environment when running NEUT
	    if [ "${SIMNAME}" == "neut" ]; then
cat >> ${SUBFILE}<<EOF
source ${NEUT_ROOT}/neutsmpl/setup.sh

EOF
            fi

            # Copy generator setup file and flux file to RAM disk for interaction generation
            if [ "${SETUP}" == "" ]; then

cat >> ${SUBFILE}<<EOF
cp ${FLUX_FULL_PATH[$idetid]} . &
cp ${NUMC_DIR}/setup/${MAXINT_FILE} . &
wait

EOF
	    fi

	# Additional header commands for ND280 software chain
	else

   	# Port-forwarding if experiencing problems connecting
        # to the TRIUMF MYSQL database from within a node
cat >> ${SUBFILE}<<EOF
ssh -N -f -L 11111:t2kcaldb.triumf.ca:3306 datamover1 
ssh -N -f -L 22222:neut08.triumf.ca:3306 datamover1
export ENV_TSQL_URL=mysql://127.0.0.1:11111/nd280calib
export ENV_GSC_TSQL_URL=mysql://127.0.0.1:22222/t2kgscND280

EOF

        fi
	
    fi
    
}



function sub_job {

    # Save current location
    CURR_DIR=`pwd`

    # Make job submission script executable
    chmod +x ${SUBFILE}

    # Must cd into spool directory for PBS to copy output to
    if [ "${SETUP}" == "setup" ]; then
	cd ${PRODUCTION_DIR}/spool
    else
	cd ${HERE}/spool
    fi

    # THE ACTUAL SUBMISSION OMG
    if [ ${SUBJOB_FOR_REAL} -eq 1 ]; then
	echo "Real qsub ${SUBFILE}"
	qsub ${SUBFILE}
    else
	echo "Fake qsub ${SUBFILE}"
    fi
    echo ""

    # Return to previous directory
    cd ${CURR_DIR}

    # Reset processor count and increment batch number
    PROCS=0
    let BATCH=$BATCH+1

}



function sub_unfilled_job {

    # Leftover script (unfilled node)
    if [ $PROCS -ne "0" ]; then

cat >> ${SUBFILE}<<EOF
wait

EOF
        echo ""
        echo "Warning unfilled node: ${PROCS}/${NPROCS} processors, Batch #${BATCH}"

	sub_job

    fi

}



function chk_err_file {
    # Check for file that contains list of run/subruns with processing error
    if [ ! -e "${ERR_FILE}" ]; then
	echo "Warning: ${ERR_FILE} does not exist. Did you run chkLogs.sh?"
	echo "         Selecting Yes will copy all files assuming there were no processing errors."
	select yn in "Yes" "No"; do
	    case $yn in
		Yes ) ERR_FILE=""; break;;
		No ) echo "Please run chkLogs.sh to find any incomplete files."; exit;;
	    esac
	done
    fi
}


