#!/bin/bash

# From http://stackoverflow.com/a/246128/1876449
# ----------------------------------------------
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTDIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# -----------------
# Detect usual bits
# -----------------

ARCH=$(uname -s)
MACH=$(uname -m)
NODE=$(uname -n)

# ------------------------------
# Define an in-place sed command
# Because Mac sed is stupid old,
# use gsed if found.
# ------------------------------

if [[ $ARCH == Darwin ]]
then
   if [[ $(command -v gsed) ]]
   then
      echo "Found gsed on macOS. Good job! You are smart!"
      SED="$(command -v gsed) -i "
   else
      echo "It is recommended to use GNU sed since macOS default"
      echo "sed is a useless BSD variant. Consider installing"
      echo "GNU sed from a packager like Homebrew:"
      echo "  brew install gnu-sed"
      SED="$(command -v sed) -i.macbak "
   fi 
else
   SED="$(command -v sed) -i "
fi

# -----
# Usage
# -----

usage ()
{
   echo "Usage: $0 [APS] [TRACE] [OPSB] [NXY] [PPN] [DT] [IPIN] [MBIND] [OPSBIO] [1HR] [3HR] [6HR] [0DY] [2DY] [5DY] [10DY] [1MO] [MEM] [MINMAX] [CHOU] [RRTMG] [HEMCO] [GAAS] [REPLAY] [NOREPLAY] [DASMODE] [BENCH] [HIGH] [HAS] [BRO] [SKY] [PGI] [SATSIM] [IOS] [NRL] [STOCH] [NOHIS] [HISTORY] [TINY] [MEDIUM] [HUGE] [RRTMG_SW] [RRTMG_LW] [LINK]  [G40] [H50] [I10] [I30] [J10] [NC4] [BIN] [WW3] [MIC] [DAS] [POLICE]"
   echo ""
   echo " Common Well-Tested Options "
   echo " ========================== "
   echo "        APS: Make experiment with APS report"
   echo "        IPM: Make experiment with IPM report"
   echo "      TRACE: Turn on ESMF tracing"
   echo "        LOG: Turn on ESMF logging"
   echo "        NXY: Set NX, NY, ntasks"
   echo "        PPN: Tasks per node"
   echo "       JCAP: JCAP number"
   echo "         DT: Set heartbeat DT"
   echo "       IPIN: Intel MPI pinning"
   echo "      MBIND: Use mbind.x at NAS"
   echo "      SHMEM: Use shmem"
   echo "     SKYOPS: Make an ops-like sky experiment"
   echo "       OPSB: Make an ops-bench experiment"
   echo "     OPSBIO: Make an ops-bench experiment with IOserver"
   echo "    OPSPORT: Make a portable ops-bench experiment"
   echo "        1HR: Make a one-hour experiment"
   echo "        3HR: Make a three-hour experiment"
   echo "        6HR: Make a six-hour experiment"
   echo "        0DY: Make a zero-day experiment"
   echo "        2DY: Make a two-day experiment"
   echo "        5DY: Make a five-day experiment"
   echo "       10DY: Make a ten-day experiment"
   echo "        1MO: Use single-moment moist"
   echo "        2MO: Use two-moment moist"
   echo "        MEM: Add memory stats"
   echo "     MINMAX: Use minmax timers"
   echo "       ZEIT: Use zeit timers"
   echo "       CHOU: Enable both shortwave and longwave Chou code"
   echo "      RRTMG: Enable both shortwave and longwave RRTMG code"
   echo "      HEMCO: Enable HEMCO"
   echo "       GAAS: Enable GAAS"
   echo "     REPLAY: Turn on regular replay"
   echo "   NOREPLAY: Turn on regular replay, but don't replay"
   echo "    DASMODE: Turn on DASMODE"
   echo "      BENCH: Use benchmark qos"
   echo "       HIGH: Use high qos"
   echo "    NOCHECK: Don't checkpoint"
   echo "        HAS: Turn on Haswell at NAS"
   echo "        BRO: Turn on Broadwell at NAS"
   echo "        SKY: Turn on Skylake at NAS (Electra)"
   echo "        PGI: Use 12 cores per node"
   echo "     SATSIM: Turn on SATSIM (ISCCP)"
   echo "        IOS: Turn on IOSERVER"
   echo "        NRL: Turn on NRLSSI2"
   echo "      STOCH: Turn on stoch physics"
   echo ""
   echo " Less Common Options "
   echo " =================== "
   echo "      NOHIS: Enables no HISTORY output"
   echo "    HISTORY: Enables smaller HISTORY.rc"
   echo "       TINY: Use minimal BCs"
   echo "     MEDIUM: Use medium BCs"
   echo "       HUGE: Use huge BCs"
   echo "   RRTMG_SW: Enable shortwave RRTMG code"
   echo "   RRTMG_LW: Enable longwave RRTMG code"
   echo "       LINK: Link restarts in scratch"
   echo ""
   echo " Rare or Obsolete Options (Use at your own risk) "
   echo " =============================================== "
   echo "        G40: Use Ganymed-4_0 directories"
   echo "        H50: Use Heracles-5_0 directories"
   echo "        I10: Use Icarus-1_0 restarts"
   echo "        I30: Use Icarus-3_0 restarts"
   echo "        J10: Use Jason-1_0 restarts"
   echo "        NC4: Convert to use NC4 restarts"
   echo "        BIN: Convert to use binary restarts"
   echo "        WW3: Turn on WAVEWATCH III capabilities"
   echo "        MIC: Enable MIC code"
   echo "     POLICE: Enables PoliceMe functionality"
   echo "        DAS: Add DAS SBATCH pragmas"
   echo ""
   echo " Note: Lowercase also allowed"
}

# ------------------------------
# Process command line arguments
# ------------------------------

# Set defaults
# ------------

USEG40=FALSE
USEH50=FALSE
USEI10=FALSE
USEI30=FALSE
USEJ10=FALSE
APS=FALSE
IPM=FALSE
TRACE=FALSE
LOG=FALSE
NXY=FALSE
PPN=FALSE
JCAP=FALSE
DT=FALSE
IPIN=FALSE
MBIND=FALSE
SHMEM=FALSE
SKYOPS=FALSE
OPSB=FALSE
OPSBIO=FALSE
OPSPORT=FALSE
ONEHOUR=FALSE
THREEHOUR=FALSE
SIXHOUR=FALSE
ZERODAY=FALSE
TWODAY=FALSE
FIVEDAY=FALSE
TENDAY=FALSE
ONEMO=FALSE
TWOMO=FALSE
NC4=FALSE
BIN=FALSE
WW3=FALSE
MEM=FALSE
MINMAX=FALSE
ZEIT=FALSE
TINY=FALSE
MEDIUM=FALSE
HUGE=FALSE
MIC=FALSE
CHOU=FALSE
RRTMG=FALSE
RRTMG_SW=FALSE
RRTMG_LW=FALSE
NOHIS=FALSE
HISTORY=FALSE
POLICE=FALSE
LINK=FALSE
DAS=FALSE
BENCH=FALSE
HIGH=FALSE
NOCHECK=FALSE
REPLAY=FALSE
NOREPLAY=FALSE
DASMODE=FALSE
HAS=FALSE
BRO=FALSE
SKY=FALSE
PGI=FALSE
SATSIM=FALSE
IOS=FALSE
NRL=FALSE
STOCH=FALSE
HEMCO=FALSE
GAAS=FALSE

while [ "${1+defined}" ]
do
   case "$1" in
      "G40" | "g40")
         USEG40=TRUE
         shift
         ;;
      "H50" | "h50")
         USEH50=TRUE
         shift
         ;;
      "I10" | "i10")
         USEI10=TRUE
         shift
         ;;
      "I30" | "i30")
         USEI30=TRUE
         shift
         ;;
      "J10" | "j10")
         USEJ10=TRUE
         shift
         ;;
      "APS" | "aps")
         APS=TRUE
         shift
         ;;
      "IPM" | "ipm")
         IPM=TRUE
         shift
         ;;
      "TRACE" | "trace")
         TRACE=TRUE
         shift
         ;;
      "LOG" | "log")
         LOG=TRUE
         shift
         ;;
      "NXY" | "nxy")
         NXY=TRUE
         shift
         NEWNX=$1
         shift
         NEWNY=$1
         shift
         NEWNTASKS=$((NEWNX*NEWNY))
         ;;
      "PPN" | "ppn")
         PPN=TRUE
         shift
         TASKS_PER_NODE=$1
         shift
         ;;
      "JCAP" | "jcap")
         JCAP=TRUE
         shift
         JCAP_VALUE=$1
         shift
         ;;
      "DT" | "dt")
         DT=TRUE
         shift
         NEWDT=$1
         shift
         ;;
      "IPIN" | "ipin")
         IPIN=TRUE
         shift
         ;;
      "MBIND" | "mbind")
         MBIND=TRUE
         shift
         ;;
      "SHMEM" | "shmem")
         SHMEM=TRUE
         shift
         ;;
      "SKYOPS" | "skyops")
         SKYOPS=TRUE
         shift
         ;;
      "OPSB" | "opsb")
         OPSB=TRUE
         shift
         ;;
      "OPSBIO" | "opsbio")
         OPSBIO=TRUE
         shift
         ;;
      "OPSPORT" | "opsport")
         OPSPORT=TRUE
         shift
         ;;
      "1HR" | "1hr")
         ONEHOUR=TRUE
         shift
         ;;
      "3HR" | "3hr")
         THREEHOUR=TRUE
         shift
         ;;
      "6HR" | "6hr")
         SIXHOUR=TRUE
         shift
         ;;
      "0DY" | "0dy")
         ZERODAY=TRUE
         shift
         ;;
      "2DY" | "2dy")
         TWODAY=TRUE
         shift
         ;;
      "5DY" | "5dy")
         FIVEDAY=TRUE
         shift
         ;;
      "10DY" | "10dy")
         TENDAY=TRUE
         shift
         ;;
      "1MO" | "1mo")
         ONEMO=TRUE
         shift
         ;;
      "2MO" | "2mo")
         TWOMO=TRUE
         shift
         ;;
      "NC4" | "nc4")
         NC4=TRUE
         shift
         ;;
      "BIN" | "bin")
         BIN=TRUE
         shift
         ;;
      "MEM" | "mem")
         MEM=TRUE
         shift
         ;;
      "MINMAX" | "minmax")
         MINMAX=TRUE
         shift
         ;;
      "ZEIT" | "zeit")
         ZEIT=TRUE
         shift
         ;;
      "TINY" | "tiny" | "MIN" | "min")
         TINY=TRUE
         shift
         ;;
      "MEDIUM" | "medium" | "MED" | "med")
         MEDIUM=TRUE
         shift
         ;;
      "HUGE" | "huge" )
         HUGE=TRUE
         shift
         ;;
      "WW3" | "ww3")
         WW3=TRUE
         shift
         ;;
      "MIC" | "mic")
         MIC=TRUE
         shift
         ;;
      "CHOU" | "chou")
         CHOU=TRUE
         shift
         ;;
      "RRTMG" | "rrtmg")
         RRTMG=TRUE
         shift
         ;;
      "RRTMG_SW" | "rrtmg_sw")
         RRTMG_SW=TRUE
         shift
         ;;
      "RRTMG_LW" | "rrtmg_lw")
         RRTMG_LW=TRUE
         shift
         ;;
      "NOHIS" | "nohis")
         NOHIS=TRUE
         shift
         ;;
      "HIS" | "his" | "HISTORY" | "history" )
         HISTORY=TRUE
         shift
         ;;
      "POL" | "pol" | "POLICE" | "police" )
         POLICE=TRUE
         shift
         ;;
      "LINK" | "link" )
         LINK=TRUE
         shift
         ;;
      "DAS" | "das" )
         DAS=TRUE
         shift
         ;;
      "BENCH" | "bench" )
         BENCH=TRUE
         shift
         ;;
      "HIGH" | "high" )
         HIGH=TRUE
         shift
         ;;
      "NOCHECK" | "nocheck" )
         NOCHECK=TRUE
         shift
         ;;
      "REPLAY" | "replay" )
         REPLAY=TRUE
         shift
         ;;
      "NOREPLAY" | "noreplay" )
         NOREPLAY=TRUE
         shift
         ;;
      "DASMODE" | "dasmode" )
         DASMODE=TRUE
         shift
         ;;
      "HAS" | "has" )
         HAS=TRUE
         shift
         ;;
      "BRO" | "bro" )
         BRO=TRUE
         shift
         ;;
      "SKY" | "sky" )
         SKY=TRUE
         shift
         ;;
      "PGI" | "pgi" )
         PGI=TRUE
         shift
         ;;
      "HEMCO" | "hemco" )
         HEMCO=TRUE
         shift
         ;;
      "GAAS" | "gaas" )
         GAAS=TRUE
         shift
         ;;
      "SATSIM" | "satsim" )
         SATSIM=TRUE
         shift
         ;;
      "IOS" | "ios" )
         IOS=TRUE
         shift
         ;;
      "NRL" | "nrl" )
         NRL=TRUE
         shift
         ;;
      "STOCH" | "stoch" )
         STOCH=TRUE
         shift
         ;;
      -h | --help)
         usage
         exit 0
         ;;
      *)
         echo "Unknown option: $1"
         echo ""
         usage
         exit 1
         ;;
   esac
done

# -----------------------------------
# If we find Tiny or Huge in the path
# of this script, assume we want to
# run the Tiny or HUGE option 
# -----------------------------------

if [[ $SCRIPTDIR =~ "Tiny" ]]
then
   TINY=TRUE
fi

if [[ $SCRIPTDIR =~ "Medium" ]]
then
   MEDIUM=TRUE
fi

if [[ $SCRIPTDIR =~ "Huge" ]]
then
   HUGE=TRUE
fi

if [[ $SKYOPS == TRUE ]] 
then
   SKY=TRUE
fi

if [[ $OPSB == TRUE ]] 
then
   BENCH=TRUE
   DASMODE=TRUE
   #NRL=TRUE
   #RRTMG=TRUE
   LINK=TRUE
   MINMAX=TRUE
   #MEM=TRUE
fi

if [[ $OPSBIO == TRUE ]] 
then
   BENCH=TRUE
   DASMODE=TRUE
   #NRL=TRUE
   #RRTMG=TRUE
   LINK=TRUE
   MINMAX=TRUE
   #MEM=TRUE
fi

if [[ $OPSPORT == TRUE ]] 
then
   BENCH=TRUE
   MINMAX=TRUE
   MEM=TRUE
   FIVEDAY=TRUE
   HUGE=TRUE
fi

if [[ $NRL == TRUE ]]
then
   RRTMG=TRUE
fi

if [[ $USEG40 == TRUE ]] 
then
   echo "Using Ganymed-4_0 directories"
   BCDIRNAME="G40"
elif [[ $USEH50 == TRUE ]] 
then
   echo "Using Heracles-5_0 directories"
   BCDIRNAME="H50"
elif [[ $USEI10 == TRUE ]] 
then
   echo "Using Icarus-1_0 directories"
   BCDIRNAME="I10"
elif [[ $USEI30 == TRUE ]] 
then
   echo "Using Icarus-3_0 directories"
   BCDIRNAME="I30"
elif [[ $USEJ10 == TRUE ]] 
then
   echo "Using Jason-1_0 directories"
   BCDIRNAME="J10"
else
   echo "Using Jason-1_0 directories"
   BCDIRNAME="J10"
fi

if [[ $SIXHOUR == TRUE ]] 
then
   echo "Making six-hour experiment with "
elif [[ $ONEHOUR == TRUE ]] 
then
   echo "Making one-hour experiment with "
elif [[ $THREEHOUR == TRUE ]] 
then
   echo "Making three-hour experiment with "
elif [[ $TWODAY == TRUE ]]
then
   echo "Making two-day experiment with "
elif [[ $ZERODAY == TRUE ]]
then
   echo "Making zero-day experiment with "
elif [[ $TEN == TRUE ]]
then
   echo "Making ten-day experiment with "
else
   echo "Making one-day experiment "
fi

if [[ $NC4 == TRUE && $BIN == TRUE ]]
then
   echo "You can't have both NC4 and BIN set to true"
   exit 9
fi


if [[ ! -z $OS12 ]]
then
   SKY=TRUE
fi


if [[ $USEG40 == TRUE ]]; then echo "     USEG40: $USEG40"; fi
if [[ $USEH50 == TRUE ]]; then echo "     USEH50: $USEH50"; fi
if [[ $USEI10 == TRUE ]]; then echo "     USEI10: $USEI10"; fi
if [[ $USEI30 == TRUE ]]; then echo "     USEI30: $USEI30"; fi
if [[ $USEJ10 == TRUE ]]; then echo "     USEJ10: $USEJ10"; fi
if [[ $APS == TRUE ]]; then echo "       APS: $APS"; fi
if [[ $IPM == TRUE ]]; then echo "       IPM: $IPM"; fi
if [[ $TRACE == TRUE ]]; then echo "       TRACE: $TRACE"; fi
if [[ $LOG == TRUE ]]; then echo "       LOG: $LOG"; fi
if [[ $NXY == TRUE ]]; then echo "       NX: $NEWNX  NY: $NEWNY NTASKS: $NEWNTASKS"; fi
if [[ $PPN == TRUE ]]; then echo "       PPN: $TASKS_PER_NODE"; fi
if [[ $JCAP == TRUE ]]; then echo "       JCAP: $JCAP_VALUE"; fi
if [[ $DT == TRUE ]]; then echo "       DT: $NEWDT"; fi
if [[ $IPIN == TRUE ]]; then echo "       IPIN: $IPIN"; fi
if [[ $MBIND == TRUE ]]; then echo "       MBIND: $MBIND"; fi
if [[ $SHMEM == TRUE ]]; then echo "       SHMEM: $SHMEM"; fi
if [[ $SKYOPS == TRUE ]]; then echo "       SKYOPS: $SKYOPS"; fi
if [[ $OPSB == TRUE ]]; then echo "       OPSB: $OPSB"; fi
if [[ $OPSBIO == TRUE ]]; then echo "     OPSBIO: $OPSBIO"; fi
if [[ $OPSPORT == TRUE ]]; then echo "    OPSPORT: $OPSPORT"; fi
if [[ $ONEHOUR == TRUE ]]; then echo "        1HR: $ONEHOUR"; fi
if [[ $THREEHOUR == TRUE ]]; then echo "        3HR: $THREEHOUR"; fi
if [[ $SIXHOUR == TRUE ]]; then echo "        6HR: $SIXHOUR"; fi
if [[ $ZERODAY == TRUE ]]; then echo "        0DY: $ZERODAY"; fi
if [[ $TWODAY == TRUE ]]; then echo "        2DY: $TWODAY"; fi
if [[ $FIVEDAY == TRUE ]]; then echo "        5DY: $FIVEDAY"; fi
if [[ $TENDAY == TRUE ]]; then echo "       10DY: $TENDAY"; fi
if [[ $ONEMO == TRUE ]]; then echo "        1MO: $ONEMO"; fi
if [[ $TWOMO == TRUE ]]; then echo "        2MO: $TWOMO"; fi
if [[ $NC4 == TRUE ]]; then echo "        NC4: $NC4"; fi
if [[ $BIN == TRUE ]]; then echo "        BIN: $BIN"; fi
if [[ $MEM == TRUE ]]; then echo "        MEM: $MEM"; fi
if [[ $MINMAX == TRUE ]]; then echo "     MINMAX: $MINMAX"; fi
if [[ $ZEIT == TRUE ]]; then echo "     ZEIT: $ZEIT"; fi
if [[ $TINY == TRUE ]]; then echo "       TINY: $TINY"; fi
if [[ $MEDIUM == TRUE ]]; then echo "       MEDIUM: $MEDIUM"; fi
if [[ $HUGE == TRUE ]]; then echo "       HUGE: $HUGE"; fi
if [[ $WW3 == TRUE ]]; then echo "        WW3: $WW3"; fi
if [[ $MIC == TRUE ]]; then echo "        MIC: $MIC"; fi
if [[ $CHOU == TRUE ]]; then echo "        CHOU: $CHOU"; fi
if [[ $RRTMG == TRUE ]]; then echo "      RRTMG: $RRTMG"; fi
if [[ $RRTMG_SW == TRUE ]]; then echo "   RRTMG_SW: $RRTMG_SW"; fi
if [[ $RRTMG_LW == TRUE ]]; then echo "   RRTMG_LW: $RRTMG_LW"; fi
if [[ $NOHIS == TRUE ]]; then echo "      NOHIS: $NOHIS"; fi
if [[ $HISTORY == TRUE ]]; then echo "    HISTORY: $HISTORY"; fi
if [[ $POLICE == TRUE ]]; then echo "     POLICE: $POLICE"; fi
if [[ $LINK == TRUE ]]; then echo "       LINK: $LINK"; fi
if [[ $DAS == TRUE ]]; then echo "        DAS: $DAS"; fi
if [[ $BENCH == TRUE ]]; then echo "      BENCH: $BENCH"; fi
if [[ $HIGH == TRUE ]]; then echo "      HIGH: $HIGH"; fi
if [[ $NOCHECK == TRUE ]]; then echo "      NOCHECK: $NOCHECK"; fi
if [[ $REPLAY == TRUE ]]; then echo "     REPLAY: $REPLAY"; fi
if [[ $NOREPLAY == TRUE ]]; then echo "     NOREPLAY: $NOREPLAY"; fi
if [[ $DASMODE == TRUE ]]; then echo "     DASMODE: $DASMODE"; fi
if [[ $HAS == TRUE ]]; then echo "        HAS: $HAS"; fi
if [[ $BRO == TRUE ]]; then echo "        BRO: $BRO"; fi
if [[ $SKY == TRUE ]]; then echo "        SKY: $SKY"; fi
if [[ $PGI == TRUE ]]; then echo "        PGI: $PGI"; fi
if [[ $HEMCO == TRUE ]]; then echo "      HEMCO: $HEMCO"; fi
if [[ $GAAS == TRUE ]]; then echo "      GAAS: $GAAS"; fi
if [[ $SATSIM == TRUE ]]; then echo "     SATSIM: $SATSIM"; fi
if [[ $IOS == TRUE ]]; then echo "     IOS: $IOS"; fi
if [[ $NRL == TRUE ]]; then echo "     NRL: $NRL"; fi
if [[ $STOCH == TRUE ]]; then echo "     STOCH: $STOCH"; fi
echo ""

# -------------------
# Locate where we are
# -------------------

NODENAME=$(uname -n)

if [[ $NODENAME == discover* || $NODENAME == dali* || $NODENAME == warp* || $NODENAME == borg* ]]
then
   SITE=NCCS

   COLORDIFF=/home/mathomp4/bin/colordiff
   UTIL_DIR=/discover/nobackup/mathomp4
   PBZIP2=/home/mathomp4/bin/pbzip2

   TAREXEC=tar
   CPEXEC=cp

elif [[ $NODENAME == pfe* || $NODENAME == r[0-9]*i[0-9]*n[0-9]* || $NODENAME == bridge* || $NODENAME == maia* ]]
then
   SITE=NAS

   COLORDIFF=/nobackup/gmao_SIteam/Utilities/bin/colordiff
   UTIL_DIR=/nobackup/gmao_SIteam/ModelData
   PBZIP2=/usr/bin/pbzip2

   TAREXEC=mtar
   CPEXEC='mcp -a'

else
   SITE=DESKTOP

   if [[ $ARCH == Darwin ]]
   then
      COLORDIFF=/usr/local/bin/colordiff
      UTIL_DIR=/Users/mathomp4/ModelData
      PBZIP2=/usr/local/bin/pbzip2
   else
      COLORDIFF=/ford1/share/gmao_SIteam/Utilities/bin/colordiff
      UTIL_DIR=/ford1/share/gmao_SIteam/ModelData
      PBZIP2=/ford1/share/gmao_SIteam/Utilities/bin/pbzip2
   fi

   TAREXEC=tar
   CPEXEC=cp

fi

MIN_DIR=$UTIL_DIR/TinyBCs-$BCDIRNAME
MIN_BCS_DIR=$MIN_DIR/bcs
MIN_SST_DIR=$MIN_DIR/sst
MIN_CHM_DIR=$MIN_DIR/chem
MIN_MIE_DIR=$MIN_DIR/chem/g5chem/x
MIN_AERO_DIR=$MIN_CHM_DIR/g5chem/L72/aero_clm
MIN_RESTART_DIR=$MIN_DIR/rs

MED_DIR=$UTIL_DIR/MediumBCs-$BCDIRNAME
MED_BCS_DIR=$MED_DIR/bcs
MED_SST_DIR=$MED_DIR/sst
MED_CHM_DIR=$MED_DIR/chem
MED_MIE_DIR=$MED_DIR/chem/g5chem/x
MED_AERO_DIR=$MED_CHM_DIR/g5chem/L72/aero_clm
MED_RESTART_DIR=$MED_DIR/rs

HUGE_DIR=$UTIL_DIR/HugeBCs-$BCDIRNAME
HUGE_BCS_DIR=$HUGE_DIR/bcs
HUGE_SST_DIR=$HUGE_DIR/sst
HUGE_CHM_DIR=$HUGE_DIR/chem
HUGE_MIE_DIR=$HUGE_DIR/chem/g5chem/x
HUGE_AERO_DIR=$HUGE_CHM_DIR/g5chem/L72/aero_clm
HUGE_RESTART_DIR=$HUGE_DIR/rs

RESTARTS_H10_DIR=Restarts-H10
RESTARTS_I10_DIR=Restarts-I10
RESTARTS_I30_DIR=Restarts-I30
RESTARTS_J10_DIR=Restarts-J10

# ------------------------------------------
# Broadwell only works at NAS. Die otherwise
# ------------------------------------------

#if [[ $HAS == TRUE && $SITE != NAS ]]
#then
   #echo "Detected site: $SITE and HAS: $HAS"
   #echo "HAS only works at NAS"
   #exit 401
#fi

if [[ $BRO == TRUE && $SITE != NAS ]]
then
   echo "Detected site: $SITE and BRO: $BRO"
   echo "BRO only works at NAS"
   exit 402
fi

#if [[ $SKY == TRUE && $SITE != NAS ]]
#then
   #echo "Detected site: $SITE and SKY: $SKY"
   #echo "SKY only works at NAS"
   #exit 403
#fi

# ---------------
# Local Functions
# ---------------

restore_save ()
{
   if [ -e $1.save ]
   then
      echo "Restoring $1.save to $1..."
      mv $1.save $1
   fi
}

copy_save ()
{
   if [ ! -e $1.save ]
   then
      echo "Copying $1 to $1.save..."
      $CPEXEC $1 $1.save
   fi
}

print_changes ()
{
   DIFF=$(diff "$1.save" "$1")
   if [ $? -ne 0 ]
   then
      echo "Changes made to $1:"
      $COLORDIFF $1.save $1
   fi
   echo
}

convert_rrtmg_none ()
{
   AERODIR=$UTIL_DIR/AerosolTables-$BCDIRNAME/ChouS-ChouI

   if [[ $TINY == TRUE ]]
   then
      AERODIR=$MIN_MIE_DIR/ChouS-ChouI
   elif [[ $MEDIUM == TRUE ]]
   then
      AERODIR=$MED_MIE_DIR/ChouS-ChouI
   elif [[ $HUGE == TRUE ]]
   then
      AERODIR=$HUGE_MIE_DIR/ChouS-ChouI
   fi

   $SED -r -e "/^ *USE_RRTMG_IRRAD/d" AGCM.rc
   $SED -r -e "/^ *USE_RRTMG_SORAD/d" AGCM.rc
   $SED -r -e "/^ *ISOLVAR/d" AGCM.rc
   $SED -r -e "/^ *USE_NRLSSI2/d" AGCM.rc
   $SED -r -e "/^ *SOLAR_CYCLE_FILE_NAME/d" AGCM.rc

   $SED -r -e "/^ *DU_OPTICS:/  s#ExtData/.*/x/opticsBands_DU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_DU.ChouS-ChouI.v\1.nc#" \
           -e "/^ *SS_OPTICS:/  s#ExtData/.*/x/opticsBands_SS.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SS.ChouS-ChouI.v\1.nc#" \
           -e "/^ *SU_OPTICS:/  s#ExtData/.*/x/opticsBands_SU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SU.ChouS-ChouI.v\1.nc#" \
           -e "/^ *OC_OPTICS:/  s#ExtData/.*/x/opticsBands_OC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_OC.ChouS-ChouI.v\1.nc#" \
           -e "/^ *BC_OPTICS:/  s#ExtData/.*/x/opticsBands_BC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BC.ChouS-ChouI.v\1.nc#" \
           -e "/^ *NI_OPTICS:/  s#ExtData/.*/x/opticsBands_NI.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_NI.ChouS-ChouI.v\1.nc#" \
           -e "/^ *BRC_OPTICS:/ s#ExtData/.*/x/opticsBands_BRC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BRC.ChouS-ChouI.v\1.nc#" AGCM.rc

   $SED -r -e "/^ *NUM_BANDS:/ s/[0-9]{2}/18/" AGCM.rc

}

convert_rrtmg_lw ()
{
   AERODIR=$UTIL_DIR/AerosolTables-$BCDIRNAME/ChouS-RRTMGI

   if [[ $TINY == TRUE ]]
   then
      AERODIR=$MIN_MIE_DIR/ChouS-RRTMGI
   elif [[ $MEDIUM == TRUE ]]
   then
      AERODIR=$MED_MIE_DIR/ChouS-RRTMGI
   elif [[ $HUGE == TRUE ]]
   then
      AERODIR=$HUGE_MIE_DIR/ChouS-RRTMGI
   fi

   $SED -r -e "/^ *USE_RRTMG_IRRAD/d" AGCM.rc
   $SED -r -e "/^ *USE_RRTMG_SORAD/d" AGCM.rc
   $SED -r -e "/^ *ISOLVAR/d" AGCM.rc
   $SED -r -e "/^ *USE_NRLSSI2/d" AGCM.rc
   $SED -r -e "/^ *SOLAR_CYCLE_FILE_NAME/d" AGCM.rc

   $SED -r -e "/^ *DU_OPTICS:/  s#ExtData/.*/x/opticsBands_DU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_DU.ChouS-RRTMGI.v\1.nc#" \
           -e "/^ *SS_OPTICS:/  s#ExtData/.*/x/opticsBands_SS.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SS.ChouS-RRTMGI.v\1.nc#" \
           -e "/^ *SU_OPTICS:/  s#ExtData/.*/x/opticsBands_SU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SU.ChouS-RRTMGI.v\1.nc#" \
           -e "/^ *OC_OPTICS:/  s#ExtData/.*/x/opticsBands_OC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_OC.ChouS-RRTMGI.v\1.nc#" \
           -e "/^ *BC_OPTICS:/  s#ExtData/.*/x/opticsBands_BC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BC.ChouS-RRTMGI.v\1.nc#" \
           -e "/^ *NI_OPTICS:/  s#ExtData/.*/x/opticsBands_NI.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_NI.ChouS-RRTMGI.v\1.nc#" \
           -e "/^ *BRC_OPTICS:/ s#ExtData/.*/x/opticsBands_BRC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BRC.ChouS-RRTMGI.v\1.nc#" AGCM.rc

   $SED -r -e "/^ *NUM_BANDS:/ s/[0-9]{2}/24/" AGCM.rc

   echo "USE_RRTMG_IRRAD: 1.0" >> AGCM.rc
}

convert_rrtmg_sw ()
{
   AERODIR=$UTIL_DIR/AerosolTables-$BCDIRNAME/RRTMGS-ChouI

   if [[ $TINY == TRUE ]]
   then
      AERODIR=$MIN_MIE_DIR/RRTMGS-ChouI
   elif [[ $MEDIUM == TRUE ]]
   then
      AERODIR=$MED_MIE_DIR/RRTMGS-ChouI
   elif [[ $HUGE == TRUE ]]
   then
      AERODIR=$HUGE_MIE_DIR/RRTMGS-ChouI
   fi

   $SED -r -e "/^ *USE_RRTMG_IRRAD/d" AGCM.rc
   $SED -r -e "/^ *USE_RRTMG_SORAD/d" AGCM.rc
   $SED -r -e "/^ *ISOLVAR/d" AGCM.rc
   $SED -r -e "/^ *USE_NRLSSI2/d" AGCM.rc
   $SED -r -e "/^ *SOLAR_CYCLE_FILE_NAME/d" AGCM.rc

   $SED -r -e "/^ *DU_OPTICS:/  s#ExtData/.*/x/opticsBands_DU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_DU.RRTMGS-ChouI.v\1.nc#" \
           -e "/^ *SS_OPTICS:/  s#ExtData/.*/x/opticsBands_SS.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SS.RRTMGS-ChouI.v\1.nc#" \
           -e "/^ *SU_OPTICS:/  s#ExtData/.*/x/opticsBands_SU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SU.RRTMGS-ChouI.v\1.nc#" \
           -e "/^ *OC_OPTICS:/  s#ExtData/.*/x/opticsBands_OC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_OC.RRTMGS-ChouI.v\1.nc#" \
           -e "/^ *BC_OPTICS:/  s#ExtData/.*/x/opticsBands_BC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BC.RRTMGS-ChouI.v\1.nc#" \
           -e "/^ *NI_OPTICS:/  s#ExtData/.*/x/opticsBands_NI.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_NI.RRTMGS-ChouI.v\1.nc#" \
           -e "/^ *BRC_OPTICS:/ s#ExtData/.*/x/opticsBands_BRC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BRC.RRTMGS-ChouI.v\1.nc#" AGCM.rc

   $SED -r -e "/^ *NUM_BANDS:/ s/[0-9]{2}/24/" AGCM.rc

   echo "USE_RRTMG_SORAD: 1.0" >> AGCM.rc
}

convert_rrtmg_swlw ()
{
   AERODIR=$UTIL_DIR/AerosolTables-$BCDIRNAME/RRTMGS-RRTMGI

   if [[ $TINY == TRUE ]]
   then
      AERODIR=$MIN_MIE_DIR/RRTMGS-RRTMGI
   elif [[ $MEDIUM == TRUE ]]
   then
      AERODIR=$MED_MIE_DIR/RRTMGS-RRTMGI
   elif [[ $HUGE == TRUE ]]
   then
      AERODIR=$HUGE_MIE_DIR/RRTMGS-RRTMGI
   fi

   $SED -r -e "/^ *USE_RRTMG_IRRAD/d" AGCM.rc
   $SED -r -e "/^ *USE_RRTMG_SORAD/d" AGCM.rc
   $SED -r -e "/^ *ISOLVAR/d" AGCM.rc
   $SED -r -e "/^ *USE_NRLSSI2/d" AGCM.rc
   $SED -r -e "/^ *SOLAR_CYCLE_FILE_NAME/d" AGCM.rc

   $SED -r -e "/^ *DU_OPTICS:/  s#ExtData/.*/x/opticsBands_DU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_DU.RRTMGS-RRTMGI.v\1.nc#" \
           -e "/^ *SS_OPTICS:/  s#ExtData/.*/x/opticsBands_SS.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SS.RRTMGS-RRTMGI.v\1.nc#" \
           -e "/^ *SU_OPTICS:/  s#ExtData/.*/x/opticsBands_SU.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_SU.RRTMGS-RRTMGI.v\1.nc#" \
           -e "/^ *OC_OPTICS:/  s#ExtData/.*/x/opticsBands_OC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_OC.RRTMGS-RRTMGI.v\1.nc#" \
           -e "/^ *BC_OPTICS:/  s#ExtData/.*/x/opticsBands_BC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BC.RRTMGS-RRTMGI.v\1.nc#" \
           -e "/^ *NI_OPTICS:/  s#ExtData/.*/x/opticsBands_NI.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_NI.RRTMGS-RRTMGI.v\1.nc#" \
           -e "/^ *BRC_OPTICS:/ s#ExtData/.*/x/opticsBands_BRC.*v([0-9]+_[0-9]+).*nc#$AERODIR/opticsBands_BRC.RRTMGS-RRTMGI.v\1.nc#" AGCM.rc

   $SED -r -e "/^ *NUM_BANDS:/ s/[0-9]{2}/30/" AGCM.rc

   echo "USE_RRTMG_IRRAD: 1.0" >> AGCM.rc
   echo "USE_RRTMG_SORAD: 1.0" >> AGCM.rc
}

# ------------------------------------
# Restore to original all edited files
# ------------------------------------

restore_save "AGCM.rc"
copy_save "AGCM.rc"

restore_save "CAP.rc"
copy_save "CAP.rc"

restore_save "HISTORY.rc"
copy_save "HISTORY.rc"

restore_save "gcm_run.j"
copy_save "gcm_run.j"

restore_save "regress/gcm_regress.j"
copy_save "regress/gcm_regress.j"

#restore_save "RC/GOCARTdata_ExtData.rc"
#copy_save "RC/GOCARTdata_ExtData.rc"

restore_save "RC/GEOS_ChemGridComp.rc"
copy_save "RC/GEOS_ChemGridComp.rc"

restore_save "RC/GAAS_GridComp.rc"
copy_save "RC/GAAS_GridComp.rc"

# -----------------------------
# Detect Atmospheric Resolution
# -----------------------------

AGCM_IM=$(grep "^ \+AGCM_IM:" AGCM.rc | awk '{print $2}')

case $AGCM_IM in
   12)
   ATMOS_RES="c12"
   ;;
   24)
   ATMOS_RES="c24"
   ;;
   48)
   ATMOS_RES="c48"
   ;;
   90)
   ATMOS_RES="c90"
   ;;
   180)
   ATMOS_RES="c180"
   ;;
   360)
   ATMOS_RES="c360"
   ;;
   720)
   ATMOS_RES="c720"
   ;;
   1000)
   ATMOS_RES="c1000"
   ;;
   1440)
   ATMOS_RES="c1440"
   ;;
   2880)
   ATMOS_RES="c2880"
   ;;
   768)
   ATMOS_RES="c768"
   if [[ $SITE == NCCS ]]
   then
      NEWBCSDIR="/discover/nobackup/projects/gmao/osse2/mathomp4/BCS_FILES/MAT/C768"
   elif [[ $SITE == NAS ]]
   then
      NEWBCSDIR="/nobackup/gmao_SIteam/ModelData/BCS_FILES/MAT/C768"
   fi
   ;;
   1536)
   ATMOS_RES="c1536"
   if [[ $SITE == NCCS ]]
   then
      NEWBCSDIR="/discover/nobackup/projects/gmao/osse2/mathomp4/BCS_FILES/MAT/C1536"
   elif [[ $SITE == NAS ]]
   then
      NEWBCSDIR="/nobackup/gmao_SIteam/ModelData/BCS_FILES/MAT/C1536"
   fi
   ;;
   3072)
   ATMOS_RES="c3072"
   if [[ $SITE == NCCS ]]
   then
      NEWBCSDIR="/discover/nobackup/projects/gmao/osse2/mathomp4/BCS_FILES/MAT/C3072"
   elif [[ $SITE == NAS ]]
   then
      NEWBCSDIR="/nobackup/gmao_SIteam/ModelData/BCS_FILES/MAT/C3072"
   fi
   ;;
   72)
   ATMOS_RES="72x46"
   ;;
   144)
   ATMOS_RES="144x91"
   ;;
   288)
   ATMOS_RES="288x181"
   ;;
   576)
   ATMOS_RES="576x361"
   ;;
   1152)
   ATMOS_RES="1152x721"
   ;;
   *)
   ATMOS_RES="UNKNOWN"
   echo "$ATMOS_RES atmospheric resolution found!"
   ;;
esac

# --------------------------------
# Detect Number of Pressure Levels
# --------------------------------

AGCM_LM=$(grep "^ \+AGCM_LM:" AGCM.rc | awk '{print $2}')

case $AGCM_LM in
   72)
   ATMOS_LEVS="72"
   ;;
   132)
   ATMOS_LEVS="132"
   ;;
   137)
   ATMOS_LEVS="137"
   ;;
   144)
   ATMOS_LEVS="144"
   ;;
   *)
   ATMOS_LEVS="UNKNOWN"
   echo "Unknown number of atmospheric levels detected: $AGCM_LM"
   exit 338
   ;;
esac

# For now, append L<levs) if we are running other than 72
# -------------------------------------------------------

if [[ $ATMOS_LEVS == 72 ]]
then
   ATMOS_RES_LEVS=$ATMOS_RES
else
   ATMOS_RES_LEVS=$ATMOS_RES-L$ATMOS_LEVS
fi

if [[ $OPSB == TRUE || $OPSBIO == TRUE ]]
then
   ATMOS_RES_LEVS=$ATMOS_RES_LEVS-OPSB
fi

# -----------------------
# Detect Ocean Resolution
# -----------------------

OGCM_GRIDNAME=$(grep "^OGCM_GRIDNAME:" AGCM.rc | awk '{print $2}')
OGCM_GRID_TYPE=$(echo ${OGCM_GRIDNAME#*-})

case $OGCM_GRID_TYPE in
   CF)
   OCEAN_RES="Ostia-CS"
   ;;
   DE)
   OGCM_IM=$(grep "^ \+OGCM_IM:" AGCM.rc | awk '{print $2}')

   case $OGCM_IM in
      360)
      OCEAN_RES="Reynolds"
      ;;
      1440)
      OCEAN_RES="MERRA-2"
      ;;
      2880)
      OCEAN_RES="Ostia"
      ;;
      *)
      OCEAN_RES="UNKNOWN"
      echo "Unknown $OCEAN_RES ocean latlon resolution found!"
      ;;
   esac
   ;;
   *)
   OCEAN_RES="UNKNOWN"
   echo "Unknown $OGCM_GRID_TYPE found!"
   ;;
esac

# --------------------
# Detect Restart Types
# --------------------

DYN_RESTART_TYPE=$(grep "^DYN_INTERNAL_RESTART_TYPE:" AGCM.rc | awk '{print $2}')

case $DYN_RESTART_TYPE in
   binary)
   RESTART_TYPE="binary"
   ;;
   pbinary)
   RESTART_TYPE="binary"
   ;;
   pnc4)
   RESTART_TYPE="nc4"
   ;;
   *)
   echo "DYN_INTERNAL_RESTART_TYPE not found. Assuming NC4"
   RESTART_TYPE="nc4"
   ;;
esac

# -----------------------------------------
# Do we want to convert to NC4 or to binary
# -----------------------------------------

if [[ $NC4 == TRUE && $RESTART_TYPE == "binary" ]]
then
   echo "NC4 requested and binary restart found. Converting to NC4"

   #$SED -e "/_rst$/ s/_rst/_rst.nc4/" \
        #-e "/_checkpoint$/ s/_checkpoint/_checkpoint.nc4/" AGCM.rc

   $SED -e "/.*_TYPE: \+pbinary$/ s/pbinary/pnc4/" \
        -e "/.*_TYPE: \+binary$/ s/binary/pnc4/" AGCM.rc

   # The above clobbers VEGDYN. Undo.
   $SED -e "/VEGDYN/ s/pnc4/binary/" AGCM.rc

   RESTART_TYPE="nc4"

elif [[ $BIN == TRUE && $RESTART_TYPE == "nc4" ]]
then
   echo "Binary requested and NC4 restart found. Converting to binary"

   #$SED -e "/_rst$/ s/_rst/_rst.nc4/" \
        #-e "/_checkpoint$/ s/_checkpoint/_checkpoint.nc4/" AGCM.rc

   $SED -e "/.*_TYPE: \+pnc4$/ s/pnc4/binary/" AGCM.rc

   # The above clobbers VEGDYN. Undo.
   #$SED -e "/VEGDYN/ s/pnc4/binary/" AGCM.rc

   RESTART_TYPE="binary"
fi


# ----------------------------------
# Do our restart files have suffixes
# ----------------------------------

FVNAME=$(grep "^DYN_INTERNAL_RESTART_FILE:" AGCM.rc | awk '{print $2}')
EXTENSION="${FVNAME##*.}"

# --------------------------
# Detect Boundary Conditions
# --------------------------

BCSDIR=$(grep "^setenv BCSDIR" gcm_run.j | awk '{print $3}')

#echo $BCSDIR

USING_4_0_BCS=$(echo $BCSDIR | grep -ow "Ganymed-4_0" )
USING_HNL_BCS=$(echo $BCSDIR | grep -ow "Heracles-NL" )
USING_Icarus_BCS=$(echo $BCSDIR | grep -ow "Icarus" )

USING_SplitSalt=$(grep OPENWATER_INTERNAL_RESTART_FILE AGCM.rc | awk '{print $1}')

#echo $USING_4_0_BCS

if [[ $USING_Icarus_BCS == "Icarus" ]]
then
   if [[ $TINY == TRUE ]]
   then
      RESTARTS_DIR=$MIN_RESTART_DIR
   elif [[ $MEDIUM == TRUE ]]
   then
      RESTARTS_DIR=$MED_RESTART_DIR
   elif [[ $HUGE == TRUE ]]
   then
      RESTARTS_DIR=$HUGE_RESTART_DIR
   elif [[ $USEI30 == TRUE ]]
   then
      RESTARTS_DIR=$UTIL_DIR/$RESTARTS_I30_DIR
   elif [[ $USEJ10 == TRUE ]]
   then
      RESTARTS_DIR=$UTIL_DIR/$RESTARTS_J10_DIR
   elif [[ $USING_SplitSalt ]]
   then
      echo "Found splitsalt using J10 restarts"
      RESTARTS_DIR=$UTIL_DIR/$RESTARTS_J10_DIR
   else
      RESTARTS_DIR=$UTIL_DIR/$RESTARTS_I30_DIR
   fi
elif [[ $USING_4_0_BCS == "Ganymed-4_0" ]]
then
   if [[ $TINY == TRUE ]]
   then
      RESTARTS_DIR=$MIN_RESTART_DIR
   elif [[ $MEDIUM == TRUE ]]
   then
      RESTARTS_DIR=$MED_RESTART_DIR
   elif [[ $HUGE == TRUE ]]
   then
      RESTARTS_DIR=$HUGE_RESTART_DIR
   else
      RESTARTS_DIR=$UTIL_DIR/$RESTARTS_H10_DIR
   fi
elif [[ $USING_HNL_BCS == "Heracles-NL" ]]
then
   RESTARTS_DIR=$UTIL_DIR/$RESTARTS_HNL_DIR
else
   echo "You seem to be using an unknown BCSDIR:"
   echo "   $BCSDIR"
   echo
   echo "This script can handle Ganymed-4_0."
   exit
fi

# -------------
# Link Restarts
# -------------

if [[ ! -e fvcore_internal_rst ]]
then
   if [[ ! -d $RESTARTS_DIR/$RESTART_TYPE/$OCEAN_RES/$ATMOS_RES_LEVS ]]
   then
      echo "FAILURE!"
      echo "Restarts of type $RESTART_TYPE for resolution $ATMOS_RES using $ATMOS_LEVS levels on ocean $OCEAN_RES do not exist in $RESTARTS_DIR"
      exit 2
   else
      echo "Linking $RESTART_TYPE restarts for resolution $ATMOS_RES using $ATMOS_LEVS levels on ocean $OCEAN_RES from $RESTARTS_DIR..."
      if [[ ${EXTENSION} == ${FVNAME} ]]
      then
         echo "Restarts have no EXTENSION..."
         ln -sv $RESTARTS_DIR/$RESTART_TYPE/$OCEAN_RES/$ATMOS_RES_LEVS/*_rst .
      else
         echo "Restarts have EXTENSION: $EXTENSION"
         if [[ -e $RESTARTS_DIR/$RESTART_TYPE/$OCEAN_RES/$ATMOS_RES_LEVS/fvcore_internal_rst.$EXTENSION ]]
         then
            ln -sv $RESTARTS_DIR/$RESTART_TYPE/$OCEAN_RES/$ATMOS_RES_LEVS/*_rst.$EXTENSION .
         else
            for file in $RESTARTS_DIR/$RESTART_TYPE/$OCEAN_RES/$ATMOS_RES_LEVS/*_rst
            do
               filename=$(basename $file)
               ln -sv $file ./$filename.$EXTENSION
            done
         fi
      fi
      if [[ ! -e cap_restart ]]
      then
         $CPEXEC $RESTARTS_DIR/$RESTART_TYPE/$OCEAN_RES/$ATMOS_RES_LEVS/cap_restart .
         if [[ ! -w cap_restart ]]
         then
            echo "cap_restart seems to be read only. For safety's sake, we make it writable"
            chmod -v u+w cap_restart
         fi
      else
         echo "cap_restart already exists. Not copying."
      fi
   fi
else
   echo "Found fvcore_internal_rst. Assuming you have needed restarts!"
fi

if [[ $WW3 == TRUE ]]
then
   echo -n "Linking WW3 mod_def.ww3 file for $OCEAN_RES"
   if [[ $OCEAN_RES == "Reynolds" ]] 
   then
      WW3_DIR=$UTIL_DIR/GridGen-v2.1/geos_1deg_latlon_grid_dateline_edge_poleline_edge
      echo " from $WW3_DIR"
      MOD_FILE=mod_def.ww3.CFL_for_30m
      ln -s $WW3_DIR/$MOD_FILE .
      ln -s $MOD_FILE mod_def.ww3
   elif [[ $OCEAN_RES == "Ostia" ]]
   then
      WW3_DIR=$UTIL_DIR/GridGen-v2.1/ostia_eighth_latlon_grid_dateline_edge_poleline_edge
      echo " from $WW3_DIR"
      MOD_FILE=mod_def.ww3.150_50_75_15
      ln -s $WW3_DIR/$MOD_FILE .
      ln -s $MOD_FILE mod_def.ww3
   elif [[ $OCEAN_RES == "MERRA-2" ]]
   then
      WW3_DIR=$UTIL_DIR/GridGen-v2.1/merra2_quart_deg_latlon_grid_dateline_edge_poleline_edge
      echo " from $WW3_DIR"
      MOD_FILE=mod_def.ww3.300_100_150_30
      ln -s $WW3_DIR/$MOD_FILE .
      ln -s $MOD_FILE mod_def.ww3
   else
      echo ""
      echo "ERROR: No WW3 mod_def.ww3 available for $OCEAN_RES"
      exit 2
   fi
fi


# ------
# CAP.rc
# ------

if [[ $SIXHOUR == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9] 000000/00000000 060000/' CAP.rc
elif [[ $ONEHOUR == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9] 000000/00000000 010000/' CAP.rc
elif [[ $THREEHOUR == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9] 000000/00000000 030000/' CAP.rc
elif [[ $TWODAY == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9]/00000002/' CAP.rc
elif [[ $ZERODAY == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9]/00000000/' CAP.rc
elif [[ $FIVEDAY == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9]/00000005/' CAP.rc
elif [[ $TENDAY == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9]/00000010/' CAP.rc
elif [[ $OPSB == TRUE || $OPSBIO == TRUE ]]
then
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9] 00/00000005 03/' CAP.rc
else
   $SED -e '/^JOB_SGMT:/ s/000000[0-9][0-9]/00000001/' CAP.rc
fi

$SED -e '/^NUM_SGMT:/ s/[0-9][0-9]*/1/' \
     -e '/^MAPL_ENABLE_MEMUTILS:/ s/NO/NO/' \
     -e '/^MAPL_ENABLE_TIMERS:/ s/NO/YES/' CAP.rc

if [[ $MEM == TRUE ]]
then
   $SED -e '/^MAPL_ENABLE_MEMUTILS:/ s/NO/YES/' CAP.rc
   $SED -e '/^MAPL_ENABLE_MEMUTILS:/ a MAPL_MEMUTILS_MODE: 1' CAP.rc
fi

if [[ $MINMAX == TRUE ]]
then
   $SED -e '/^MAPL_ENABLE_MEMUTILS:/ a MAPL_TIMER_MODE: MINMAX' CAP.rc
fi

if [[ $ZEIT == TRUE ]]
then
   $SED -e '/^MAPL_ENABLE_MEMUTILS:/ a MAPL_TIMER_MODE: ZEIT' CAP.rc
fi

if [[ $OPSB == TRUE || $OPSBIO == TRUE || $OPSPORT == TRUE || $SKYOPS == TRUE || $SHMEM == TRUE ]]
then
   $SED -e '/^USE_SHMEM:/ s/0/1/' CAP.rc
fi

if [[ $OPSBIO == TRUE ]]
then
   $SED -e '/^USE_SHMEM:/ a USE_IOSERVER: 1' CAP.rc
fi

if [[ $IOS == TRUE ]]
then
   $SED -e '/^USE_SHMEM:/ a USE_IOSERVER: 1' CAP.rc
fi

if [[ $DT == TRUE ]]
then
   $SED -e "/^HEARTBEAT_DT:/ s/[0-9][0-9]*/$NEWDT/" CAP.rc
fi

$SED -e '/^CoresPerNode:/ d' CAP.rc

print_changes "CAP.rc"

# ----------
# HISTORY.rc
# ----------

if [[ $HISTORY == TRUE ]]
then

   # This command should add a # to all lines that have geosgcm
   # or tavg with spaces at the beginning between COLLECTIONS and ::

   $SED -e "/^COLLECTIONS:/,/.*  ::/ {
                                     /^ .*geosgcm/ s/ /#/
                                     /^ .*tavg/    s/ /#/}" HISTORY.rc

fi

if [[ $NOHIS == TRUE ]]
then

   # This command deletes all lines that have geosgcm
   # or tavg with spaces (and start with #)
   # at the beginning between COLLECTIONS and ::

   $SED -e "/^COLLECTIONS:/,/.*  ::/ {
                                     /^ .*geosgcm/ d
                                     /^# .*geosgcm/ d
                                     /^ .*tavg/    d}" HISTORY.rc

   # Now we take care of the collection that is on the same line
   # as COLLECTIONS is

   $SED -e "/^COLLECTIONS:/ s/\(COLLECTIONS:\).*/\1/" HISTORY.rc
fi


#if [[ $TINY == TRUE ]]
#then

   #echo "Since we turn off TR, tracer collection cannot run"

   #$SED -e "/ *'geosgcm_tracer'/ s/ /#/" HISTORY.rc
#fi

if [[ $SATSIM == TRUE ]]
then

   echo "Enabling ISCCP collection"

   $SED -e "/^#.*'geosgcm_isccp'/ s/#//" HISTORY.rc
fi

if [[ $OPSB == TRUE || $OPSBIO == TRUE ]]
then
   $CPEXEC -v $SCRIPTDIR/HISTORY.rc.f521_fp_20181018_21z.for20181014 HISTORY.rc
   echo
   echo "HISTORY.rc has been replaced with OPS history!!!"
   echo
   if [[ $OPSBIO == TRUE ]]
   then
      echo "Enabling CFIOasync"

      $SED -e "s/CFIO/CFIOasync/" HISTORY.rc
   fi
else
   print_changes "HISTORY.rc"
fi

# ---------
# gcm_run.j
# ---------

$SED -e '/^echo GEOSgcm/ a exit' \
     -e '/^#SBATCH --account/ a #SBATCH --mail-type=ALL' \
     -e '/^#SBATCH -A / a #SBATCH --mail-type=ALL' \
     -e '/^#PBS -W group_list/ a #SBATCH --mail-type=ALL' \
     -e '/numrs == 0/ s/== 0/== 1/ ' gcm_run.j

if [[ $TENDAY != TRUE && $FIVEDAY != TRUE ]]
then

   if (( $AGCM_IM < 180 ))
   then
      if [[ $ATMOS_LEVS == 132 ]]
      then
         NEWTIME=30
      else
         NEWTIME=15
      fi
      $SED -e "/^#PBS -l walltime/ s/=12:00:/=0:${NEWTIME}:/ " \
           -e "/^#PBS -l walltime/ s/=8:00:/=0:${NEWTIME}:/ " \
           -e "/^#SBATCH --time/ s/=12:00:/=0:${NEWTIME}:/ " gcm_run.j
   elif (( $AGCM_IM < 720 ))
   then
      if [[ $ATMOS_LEVS == 132 ]]
      then
         NEWTIME=2
      else
         NEWTIME=1
      fi
      $SED -e "/^#PBS -l walltime/ s/=12:/=${NEWTIME}:/ " \
           -e "/^#PBS -l walltime/ s/=8:/=${NEWTIME}:/ " \
           -e "/^#SBATCH --time/ s/=12:/=${NEWTIME}:/ " gcm_run.j
   else
      if [[ $ATMOS_LEVS == 132 ]]
      then
         NEWTIME=4
      else
         NEWTIME=2
      fi
      $SED -e "/^#PBS -l walltime/ s/=12:/=${NEWTIME}:/ " \
           -e "/^#PBS -l walltime/ s/=8:/=${NEWTIME}:/ " \
           -e "/^#SBATCH --time/ s/=12:/=${NEWTIME}:/ " gcm_run.j
   fi

fi

if [[ $LINK == TRUE ]]
then
   $SED -e "/^ *if(-e \$EXPDIR\/\$rst/ s/$CPEXEC/ln -s/" gcm_run.j
fi

if [[ $SITE == NAS ]] 
then
   $SED -e '/^#PBS -l walltime/ a #PBS -W umask=0022' \
        -e '/^#PBS -l walltime/ a #PBS -m abe' gcm_run.j
fi

if [[ $DAS == TRUE ]]
then
   $SED -e '/^#SBATCH --account/ a #SBATCH --reservation=das' \
        -e '/^#SBATCH --account/ a #SBATCH --qos=das2015' \
        -e '/^#SBATCH -A / a #SBATCH --reservation=das' \
        -e '/^#SBATCH -A/ a #SBATCH --qos=das2015' gcm_run.j
fi

if [[ $BENCH == TRUE ]]
then
   $SED -e '/^#SBATCH --account/ a #SBATCH --partition=preops' \
        -e '/^#SBATCH --account/ a #SBATCH --qos=benchmark' \
        -e '/^#SBATCH -A / a #SBATCH --partition=preops' \
        -e '/^#SBATCH -A/ a #SBATCH --qos=benchmark' gcm_run.j

   $SED -e '/^#SBATCH -A/ s/s1873/g0620/g' gcm_run.j
fi

if [[ $HIGH == TRUE ]]
then
   $SED -e '/^#SBATCH --account/ a #SBATCH --qos=high' \
        -e '/^#SBATCH -A/ a #SBATCH --qos=high' gcm_run.j
fi

if [[ $PGI == TRUE ]]
then
   $SED -e '/^#SBATCH --ntasks/ a #SBATCH --ntasks-per-node=12' gcm_run.j
fi



# Section to autoconvert PBS line

NX=$(grep '^\s*NX:' AGCM.rc | awk '{print $2}') 
NY=$(grep '^\s*NY:' AGCM.rc | awk '{print $2}') 
NPES=$(echo "$NX * $NY" | bc)

if [[ $SITE == NAS ]]
then

   if [[ $HAS == TRUE ]]
   then
      NCPUS=24
      MODEL=has
   elif [[ $BRO == TRUE ]]
   then
      NCPUS=28
      MODEL=bro
   elif [[ $SKY == TRUE ]]
   then
      NCPUS=40
      MODEL=sky_ele
   else
      # Assume sky_ele at NAS
      NCPUS=40
      MODEL=sky_ele
   fi

fi

if [[ $SITE == NCCS ]]
then

   if [[ $HAS == TRUE ]]
   then
      NCPUS=28
      MODEL=hasw
   elif [[ $SKY == TRUE ]]
   then
      NCPUS=40
      MODEL=sky
   else
      # Assume hasw at NCCS
      NCPUS=28
      MODEL=hasw
   fi
fi

if [[ $SITE == NAS ]]
then
   if [[ $HAS == TRUE || $BRO == TRUE || $SKY == TRUE ]]
   then
      NCUS=$(echo "($NPES + $NCPUS - 1)/$NCPUS" | bc)
      $SED -e "/^#PBS -l select/ s/select=.*:ncpus=.*:mpiprocs=.*:model=.*/select=$NCUS:ncpus=$NCPUS:mpiprocs=$NCPUS:model=$MODEL/" gcm_run.j
   fi
fi

if [[ $SITE == NCCS ]]
then
   if [[ $SKY == TRUE ]]
   then
      $SED -e "/^#SBATCH --constraint=/ s/hasw/sky/" gcm_run.j
   fi
fi

if [[ $SITE == NCCS ]] 
then
   if [[ $PPN == TRUE ]]
   then
      $SED -e "/^#SBATCH --ntasks=/ a #SBATCH --ntasks-per-node=$TASKS_PER_NODE" gcm_run.j
   fi

   if [[ $NXY == TRUE ]]
   then
      $SED -e "/^#SBATCH --ntasks=/ s/--ntasks=\([0-9]\+\)/--ntasks=$NEWNTASKS/" gcm_run.j
   fi
fi

if [[ $SITE == NAS ]]
then

   if [[ $NXY == TRUE ]]
   then

      if [[ $PPN == TRUE ]]
      then
         NCUS=$(echo "($NEWNTASKS + $TASKS_PER_NODE - 1)/$TASKS_PER_NODE" | bc)
         $SED -e "/^#PBS -l select/ s/select=.*:ncpus=.*:mpiprocs=.*:model=.*/select=$NCUS:ncpus=$NCPUS:mpiprocs=$TASKS_PER_NODE:model=$MODEL/" gcm_run.j
      else
         NCUS=$(echo "($NEWNTASKS + $NCPUS - 1)/$NCPUS" | bc)
         $SED -e "/^#PBS -l select/ s/select=.*:ncpus=.*:mpiprocs=.*:model=.*/select=$NCUS:ncpus=$NCPUS:mpiprocs=$NCPUS:model=$MODEL/" gcm_run.j
      fi
   else
      if [[ $PPN == TRUE ]]
      then
         NCUS=$(echo "($NPES + $TASKS_PER_NODE - 1)/$TASKS_PER_NODE" | bc)
         $SED -e "/^#PBS -l select/ s/select=.*:ncpus=.*:mpiprocs=.*:model=.*/select=$NCUS:ncpus=$NCPUS:mpiprocs=$TASKS_PER_NODE:model=$MODEL/" gcm_run.j
      fi
   fi
fi


if [[ $SITE == NAS ]]
then
   if [[ $MBIND == TRUE ]]
   then

      if [[ $PPN == TRUE ]]
      then
         $SED -e "/\$RUN_CMD \$NPES/ s#\$NPES#\$NPES /u/scicon/tools/bin/mbind.x -cs -n${TASKS_PER_NODE} -v#" gcm_run.j
      else
         $SED -e "/\$RUN_CMD \$NPES/ s#\$NPES#\$NPES /u/scicon/tools/bin/mbind.x -cs -n${NCPUS} -v#" gcm_run.j
      fi
   fi
fi

if [[ $WW3 == TRUE ]]
then
   echo "Enabling WW3 in gcm_run.j"
   $SED -e '/                             \/bin\/cp \-f/ a\
                             /bin/cp -f $EXPDIR/*.ww3 .' gcm_run.j
fi 

if [[ $POLICE == TRUE ]]
then
   echo "Enabling PoliceMe in gcm_run.j"
   $SED -e '/limit stacksize/ a\
\
if { /home/mathomp4/bin/amibatch } then \
   echo "I am running in batch mode. Executing PoliceMe" \
   mkdir -p policeme \
   /usr/local/other/policeme/policeme.exe -d policeme \
endif ' gcm_run.j
fi

if [[ $MIC == TRUE ]]
then
   echo "Enabling for MIC"
   $SED -e '/limit stacksize/ a \
\
   setenv PATH /opt/intel/mic/bin:${PATH} \
   setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/opt/intel/mic/coi/host-linux-release/lib \
 \
   unsetenv MIC_ENV_PREFIX \
   unsetenv MIC_OMP_NUM_THREADS \
   unsetenv OMP_NUM_THREADS \
   unsetenv OFFLOAD_INIT \
   unsetenv MIC_USE_2MB_BUFFERS \
   unsetenv MIC_KMP_AFFINITY \
   unsetenv MKL_MIC_ENABLE \
   unsetenv OFFLOAD_REPORT \
   unsetenv I_MPI_MIC  \
   unsetenv I_MPI_FABRIC \
   unsetenv I_MPI_DEBUG \
 \
   setenv MIC_ENV_PREFIX MIC \
   setenv MIC_OMP_NUM_THREADS 236 #multiple of 59 4 threads per core \
   setenv OMP_NUM_THREADS 16 \
   setenv OFFLOAD_INIT on_start #good for timing prepares MiC \
   setenv MIC_USE_2MB_BUFFERS 2K \
   setenv MIC_KMP_AFFINITY balanced,granularity=fine \
   setenv MKL_MIC_ENABLE 1 \
   setenv OFFLOAD_REPORT 1 \
   setenv I_MPI_MIC 1  \
   setenv I_MPI_FABRIC shm:ofa \
   setenv I_MPI_DEBUG 5 ' \
   -e '/mpirun_rsh/ s/mpirun_rsh/mpirun_rsh -export/ ' gcm_run.j
fi

if [[ $TINY == TRUE ]]
then
   echo "Using minimal boundary datasets"
   $SED -e "/^setenv BCSDIR/ s#\(setenv BCSDIR\)\(.*\)#\1   ${MIN_BCS_DIR}#" \
        -e "/^setenv SSTDIR/ s#\(setenv SSTDIR\)\(.*\)#\1   ${MIN_SST_DIR}#" \
        -e "/^setenv CHMDIR/ s#\(setenv CHMDIR\)\(.*\)#\1   ${MIN_CHM_DIR}#" \
        -e "/pchem.species/  s#1870-2097#1999-2000#"                         \
        -e "/pchem.species/  s#197902-201706#1999-2000#"                     \
        -e "/sst_1971/       s#1971-current#2000#"                           \
        -e "/fraci_1971/     s#1971-current#2000#"                               gcm_run.j
fi

if [[ $MEDIUM == TRUE ]]
then
   echo "Using minimal boundary datasets"
   $SED -e "/^setenv BCSDIR/ s#\(setenv BCSDIR\)\(.*\)#\1   ${MED_BCS_DIR}#" \
        -e "/^setenv SSTDIR/ s#\(setenv SSTDIR\)\(.*\)#\1   ${MED_SST_DIR}#" \
        -e "/^setenv CHMDIR/ s#\(setenv CHMDIR\)\(.*\)#\1   ${MED_CHM_DIR}#" \
        -e "/pchem.species/  s#1870-2097#1999-2000#"                         \
        -e "/pchem.species/  s#197902-201706#1999-2000#"                     \
        -e "/sst_1971/       s#1971-current#2000#"                           \
        -e "/fraci_1971/     s#1971-current#2000#"                               gcm_run.j
fi

if [[ $HUGE == TRUE ]]
then
   echo "Using maximal boundary datasets"
   $SED -e "/^setenv BCSDIR/ s#\(setenv BCSDIR\)\(.*\)#\1   ${HUGE_BCS_DIR}#" \
        -e "/^setenv SSTDIR/ s#\(setenv SSTDIR\)\(.*\)#\1   ${HUGE_SST_DIR}#" \
        -e "/^setenv CHMDIR/ s#\(setenv CHMDIR\)\(.*\)#\1   ${HUGE_CHM_DIR}#"    gcm_run.j
fi

if [[ ! -z $NEWBCSDIR ]]
then
   echo "Using new boundary datasets"
   $SED -e "/^setenv BCSDIR/ s#\(setenv BCSDIR\)\(.*\)#\1   ${NEWBCSDIR}#" \
        -e "/^setenv SSTDIR/ s#\(setenv SSTDIR\)\(.*\)#\1   ${NEWBCSDIR}#" gcm_run.j
fi

if [[ $SKYOPS == TRUE ]]
then
   $SED -e "/^#SBATCH --ntasks=/ s/1536/5400/" \
        -e '/^#PBS -l walltime/ s/=2:00/=00:30/ ' gcm_run.j
fi

if [[ $IPIN == TRUE ]]
then
   $SED -e '/setenv I_MPI_DAPL_UD/ a \
 \
setenv I_MPI_DEBUG 4 \
setenv I_MPI_PIN_PROCESSOR_LIST allcores:map=spread' gcm_run.j
fi

if [[ $IPM == TRUE ]]
then
   $SED -e '/setenv I_MPI_DAPL_UD/ a \
 \
setenv I_MPI_DEBUG 4 \
setenv I_MPI_STATS ipm \
setenv I_MPI_STATS_FILE $EXPDIR/stats.$SLURM_JOBID.ipm' gcm_run.j
fi

if [[ $OPSPORT == TRUE ]]
then
   $SED -e "/^#SBATCH --ntasks=/ s/1536/5400/" \
        -e "/^#SBATCH --ntasks=/ a #SBATCH --ntasks-per-node=27" \
        -e '/^#PBS -l walltime/ s/=12:/=3:/ ' \
        -e '/^#PBS -l walltime/ s/=8:/=3:/ ' \
        -e '/^#PBS -l select/ s/select=64:ncpus=24:mpiprocs=24:model=has/select=200:ncpus=27:mpiprocs=27:model=bro/' gcm_run.j

fi

if [[ $OPSB == TRUE ]]
then
   $SED -e "/^#SBATCH --ntasks=/ s/1536/1944/" \
        -e "/^#SBATCH --ntasks=/ a #SBATCH --ntasks-per-node=27" \
        -e '/^#PBS -l walltime/ s/=12:/=4:/ ' \
        -e '/^#PBS -l walltime/ s/=8:/=4:/ ' \
        -e '/^#PBS -l walltime/ s/=2:/=4:/ ' \
        -e '/^#PBS -l select/ s/select=39:ncpus=40:mpiprocs=40:model=sky_ele/select=72:ncpus=27:mpiprocs=27:model=bro/' \
        -e '/^#PBS -l select/ s/select=55:ncpus=28:mpiprocs=28:model=bro/select=72:ncpus=27:mpiprocs=27:model=bro/' gcm_run.j

   $SED -e '/^#PBS -l select/ a \
#SKYNAS#PBS -l select=54:ncpus=36:mpiprocs=36:model=sky_ele' gcm_run.j

   $SED -e "/pchem.species/ s/pchem.species.CMIP-5.1870-2097.z_91x72.nc4/pchem.species.Clim_Prod_Loss.z_721x72.nc4/" gcm_run.j
   $SED -e "/pchem.species/ s/pchem.species.CMIP-5.MERRA2OX.197902-201706.z_91x72.nc4/pchem.species.Clim_Prod_Loss.z_721x72.nc4/" gcm_run.j
fi

if [[ $OPSBIO == TRUE ]]
then
   $SED -e "/^#SBATCH --ntasks=/ s/1536/2079/" \
        -e "/^#SBATCH --ntasks=/ a #SBATCH --ntasks-per-node=27" \
        -e '/^#PBS -l walltime/ s/=12:/=4:/ ' \
        -e '/^#PBS -l walltime/ s/=8:/=4:/ ' \
        -e '/^#PBS -l walltime/ s/=2:/=4:/ ' \
        -e '/^#PBS -l select/ s/select=39:ncpus=40:mpiprocs=40:model=sky_ele/select=77:ncpus=27:mpiprocs=27:model=bro/' \
        -e '/^#PBS -l select/ s/select=55:ncpus=28:mpiprocs=28:model=bro/select=77:ncpus=27:mpiprocs=27:model=bro/' gcm_run.j
   
   $SED -e '/^#PBS -l select/ a \
#SKYNAS#PBS -l select=59:ncpus=36:mpiprocs=36:model=sky_ele' gcm_run.j

   $SED -e '/^$RUN_CMD $NPES/ s/$NPES/2079/ ' gcm_run.j
   $SED -e '/^ *$RUN_CMD $NPES/ s/$NPES/2079/ ' gcm_run.j

   $SED -e '/^$RUN_CMD 2079/ s/.*/&\n#SKYNAS&/' gcm_run.j
   $SED -e '/^ *$RUN_CMD 2079/ s/.*/&\n#SKYNAS&/' gcm_run.j

   $SED -e '/^#SKYNAS$RUN_CMD 2079/ s/2079/2124/' gcm_run.j
   $SED -e '/^#SKYNAS *$RUN_CMD 2079/ s/2079/2124/' gcm_run.j

   $SED -e "/pchem.species/ s/pchem.species.CMIP-5.1870-2097.z_91x72.nc4/pchem.species.Clim_Prod_Loss.z_721x72.nc4/" gcm_run.j
   $SED -e "/pchem.species/ s/pchem.species.CMIP-5.MERRA2OX.197902-201706.z_91x72.nc4/pchem.species.Clim_Prod_Loss.z_721x72.nc4/" gcm_run.j
fi

if [[ $APS == TRUE ]]
then

   # Load the vtune module
   # ---------------------
   $SED -e "/^ *\$RUN_CMD \$NPES/ i \
   module load tool/vtune-2018" gcm_run.j

   # Env var for mpi graph
   # ---------------------
   $SED -e "/^ *\$RUN_CMD \$NPES/ i \
   setenv MPS_STAT_LEVEL 4" gcm_run.j

   # Use aps
   # -------
   $SED -e "/^ *\$RUN_CMD \$NPES/ s/NPES/NPES aps/" gcm_run.j

   # Make the report
   # ---------------
   $SED -e "/^ *\$RUN_CMD \$NPES/ a \
   /home/mathomp4/makeAPSreport.bash" gcm_run.j

fi

if [[ $TRACE == TRUE ]]
then

   # Add ESMF tracing setenvs
   # ------------------------
   $SED -e "/^\$RUN_CMD \$NPES/ i \
   setenv ESMF_RUNTIME_TRACE ON" gcm_run.j

fi

if [[ $LOG == TRUE ]]
then

   # Add ESMF logging cmd option
   # ---------------------------
   $SED -e '/$NPES .\/GEOSgcm.x/ s/GEOSgcm.x/GEOSgcm.x --esmf_logtype multi_on_error/' gcm_run.j

fi

print_changes "gcm_run.j"

# -------
# AGCM.rc
# -------

DOING_GOCART=$(grep AERO_PROVIDER AGCM.rc | awk '{print $2}')

FOUND_BOOTSTRAP=$(grep MAPL_ENABLE_BOOTSTRAP AGCM.rc | awk '{print $1}')

#$SED -e "/^NUM_WRITERS:/ s/4/1/" AGCM.rc

if [[ "$FOUND_BOOTSTRAP" == "MAPL_ENABLE_BOOTSTRAP:" ]]
then
   $SED -e "/^MAPL_ENABLE_BOOTSTRAP:/ s/NO/YES/" AGCM.rc
else
   if [[ "$DOING_GOCART" == "GOCART" && ! -e 'gocart_internal_rst' ]]
   then
      echo "Didn't see MAPL_ENABLE_BOOTSTRAP"
      echo "For safety's sake, we bootstrap gocart"
      $SED -e '/GOCART_INTERNAL_RESTART_FILE:/ s/ gocart_internal_rst/-gocart_internal_rst/' AGCM.rc
   fi
fi

if [[ $CHOU == TRUE ]]
then
   convert_rrtmg_none 
fi

if [[ $RRTMG_LW == TRUE ]]
then
   convert_rrtmg_lw 
fi

if [[ $RRTMG_SW == TRUE ]]
then
   convert_rrtmg_sw 
fi

if [[ $RRTMG == TRUE ]]
then
   convert_rrtmg_swlw 
fi

if [[ $ONEMO == TRUE ]]
then
   echo "Enabling single-moment in AGCM.rc"
   echo "CLDMICRO: 1MOMENT" >> AGCM.rc
fi

if [[ $TWOMO == TRUE ]]
then
   echo "Enabling single-moment in AGCM.rc"
   echo "CLDMICRO: 2MOMENT" >> AGCM.rc
fi

if [[ $WW3 == TRUE ]]
then
   echo "Enabling WW3 in AGCM.rc"
   echo "USE_WW3: 1" >> AGCM.rc
   echo "WRITE_WW3_RESTART: 0" >> AGCM.rc
fi


if [[ $MIC == TRUE ]]
then
   echo "Enabling for MIC in AGCM.rc"
   echo "SOLAR_LOAD_BALANCE: 0" >> AGCM.rc
fi

if [[ $REPLAY == TRUE || $NOREPLAY == TRUE ]]
then
   echo "Turning on regular replay"
   $SED -e "/^#   REPLAY_MODE: Regular/ s/#/ /" \
        -e "/REPLAY_MODE: Regular/{n;s/#/ /}" AGCM.rc

   if [[ $SITE == NAS ]]
   then
      echo "Found NAS. Transforming REPLAY directory"
      $SED -e "/^ *REPLAY_FILE/ s#/discover/nobackup/projects/gmao/share/gmao_ops#${UTIL_DIR}#" AGCM.rc
   fi
fi

if [[ $NOREPLAY == TRUE ]]
then
   echo "Setting all replay variables to NO"
   $SED -e '/^#   REPLAY_P:  YES or NO/ c\    REPLAY_P:  NO' \
        -e '/^#   REPLAY_U:  YES or NO/ c\    REPLAY_U:  NO' \
        -e '/^#   REPLAY_V:  YES or NO/ c\    REPLAY_V:  NO' \
        -e '/^#   REPLAY_T:  YES or NO/ c\    REPLAY_T:  NO' \
        -e '/^#   REPLAY_QV: YES or NO/ c\    REPLAY_QV: NO' \
        -e '/^#   REPLAY_O3: YES or NO/ c\    REPLAY_O3: NO' \
        -e '/^#   REPLAY_TS: YES or NO/ c\    REPLAY_TS: NO' AGCM.rc
fi

if [[ $DASMODE == TRUE ]]
then
   echo "Turning on dasmode"
   $SED -e "/AGCM_IMPORT_RESTART_FILE:/ s/#/ /" AGCM.rc
fi

if [[ $TINY == TRUE || $MEDIUM == TRUE ]]
then

   # Before we turned of GOCART.data. We now provide a
   # tiny version of it in Tiny-BCs

   #$SED -e "/AERO_PROVIDER:/ s/GOCART.data  /None  /" \
        #-e "/pchem_clim_years:/ s/228/2/" AGCM.rc

   $SED -e "/pchem_clim_years:/ s/228/2/" AGCM.rc
   $SED -e "/pchem_clim_years:/ s/39/2/" AGCM.rc

fi

if [[ $NOCHECK == TRUE ]]
then
   $SED -e '/^#RECORD_FINAL:/ s/#RECORD_FINAL:  >>>RECFINL<<</RECORD_FINAL: NO/' AGCM.rc
fi

if [[ $SKYOPS == TRUE ]]
then
   $SED -e '/^ *NX:/ s/16/15/' \
        -e '/^ *NY:/ s/96/360/' AGCM.rc
fi

if [[ $OPSPORT == TRUE ]]
then
   $SED -e '/^SOLAR_DT:/ s/1800/3600/' \
        -e '/^IRRAD_DT:/ s/1800/3600/' \
        -e '/^OGCM_RUN_DT:/ s/1800/3600/' AGCM.rc

   $SED -e '/^IRRAD_DT:/ a GOCART_DT: 3600' AGCM.rc

   $SED -e '/^ *NX:/ s/16/15/' \
        -e '/^ *NY:/ s/96/360/' AGCM.rc

   $SED -e '/^#RECORD_FINAL:/ s/#RECORD_FINAL:  >>>RECFINL<<</RECORD_FINAL: NO/' \
        -e '/^#RECORD_FREQUENCY:/ s/#RECORD_FREQUENCY: 000000       000000/RECORD_FREQUENCY: 000000/' \
        -e '/^#RECORD_REF_DATE:/ s/#RECORD_REF_DATE: >>>REFDATE<<< >>>FCSDATE<<</RECORD_REF_DATE: 20150415/' \
        -e '/^#RECORD_REF_TIME:/ s/#RECORD_REF_TIME: >>>REFTIME<<< >>>FCSTIME<<</RECORD_REF_TIME: 030000/' AGCM.rc
fi

if [[ $OPSB == TRUE || $OPSBIO == TRUE ]]
then
   $SED -e '/^SOLAR_DT:/ s/1800/3600/' \
        -e '/^IRRAD_DT:/ s/1800/3600/' \
        -e '/^SATSIM_DT:/ s/1800/900/' \
        -e '/^NUM_WRITERS:/ s/1/6/' \
        -e '/^OGCM_RUN_DT:/ s/1800/3600/' AGCM.rc

   #$SED -e '/^#GOCART_DT:/ s/#GOCART/ GOCART/' AGCM.rc

   $SED -e '/^ *NX:/ s/16/18/' \
        -e '/^ *NY:/ s/96/108/' AGCM.rc

   $SED -e '/^#RECORD_FINAL:/ s/#RECORD_FINAL:  >>>RECFINL<<</RECORD_FINAL: NO/' \
        -e '/^#RECORD_FREQUENCY:/ s/#RECORD_FREQUENCY: 000000       000000/RECORD_FREQUENCY: 000000/' \
        -e '/^#RECORD_REF_DATE:/ s/#RECORD_REF_DATE: >>>REFDATE<<< >>>FCSDATE<<</RECORD_REF_DATE: 20181015/' \
        -e '/^#RECORD_REF_TIME:/ s/#RECORD_REF_TIME: >>>REFTIME<<< >>>FCSTIME<<</RECORD_REF_TIME: 030000/' AGCM.rc

   $SED -e '/^# *ASSIMILATION_CYCLE:/ s/#/ /' \
        -e '/^# *CORRECTOR_DURATION:/ s/#/ /' \
        -e '/^# *PREDICTOR_DURATION:/ s/#/ /' \
        -e '/^# *IAU_DIGITAL_FILTER:/ s/#/ /' AGCM.rc

   $SED -e '/^ *ASSIMILATION_CYCLE:/ s/nnnnnn/21600/' \
        -e '/^ *CORRECTOR_DURATION:/ s/nnnnnn/3600/' \
        -e '/^ *PREDICTOR_DURATION:/ s/nnnnnn/0/' \
        -e '/^ *IAU_DIGITAL_FILTER:/ s/YES or NO/NO/' AGCM.rc

   echo "Turning on regular replay"
   $SED -e "/^#   REPLAY_MODE: Regular/ s/#/ /" \
        -e "/REPLAY_MODE: Regular/{n;s/#/ /}" AGCM.rc

   if [[ $SITE == NCCS ]]
   then
      echo "Found NAS. Transforming REPLAY directory"
      $SED -e "/^ *REPLAY_FILE/ s#share/gmao_ops/verification/MERRA2_MEANS#osse2/mathomp4/f521_fp#" AGCM.rc
      $SED -e "/^ *REPLAY_FILE/ s#MERRA-2#f521_fp#" AGCM.rc
   fi

   if [[ $SITE == NAS ]]
   then
      echo "Found NAS. Transforming REPLAY directory"
      $SED -e "/^ *REPLAY_FILE/ s#/discover/nobackup/projects/gmao/share/gmao_ops/verification/MERRA2_MEANS#/nobackupp11/mathomp4/f521_fp#" AGCM.rc
      $SED -e "/^ *REPLAY_FILE/ s#MERRA-2#f521_fp#" AGCM.rc
   fi

   $SED -e "/^# *REPLAY_SHUTOFF:/ s/#/ /" \
        -e "/^# *REPLAY_WINDFIX:/ s/#/ /" AGCM.rc

   $SED -e "/^ *REPLAY_WINDFIX:/ s/YES/NO/" AGCM.rc

   $SED -e "/^ *OX_RELAXTIME:/ s/259200./0.00/" AGCM.rc
   $SED -e "/^ *OX_RELAXTIME:/ s/^/#/" AGCM.rc
   $SED -e "/^ *pchem_clim_years:/ s/39/1/" AGCM.rc
   $SED -e "/^ *pchem_clim_years:/ s/228/1/" AGCM.rc

   $SED -e "/^ANA_GRIDNAME:/ s/720x4320/360x2160/" AGCM.rc

   $SED -e "/^# *AIAU_IMPORT_RESTART_FILE/ c\    AIAU_IMPORT_RESTART_FILE: aiau_import_rst" AGCM.rc
   $SED -e "/^# *AIAU_IMPORT_CHECKPOINT_FILE/ c\    AIAU_IMPORT_CHECKPOINT_FILE: aiau_import_checkpoint" AGCM.rc

fi

if [[ $OPSPORT == TRUE ]]
then
   $SED -e '/^RECORD_REF_DATE:/ s/2015/2000/' AGCM.rc
fi

if [[ $SATSIM == TRUE || $OPSB == TRUE || $OPSBIO == TRUE ]]
then
   $SED -e '/^USE_SATSIM_ISCCP: / s/0/1/' AGCM.rc
   $SED -e '/^USE_SATSIM_MODIS: / s/0/1/' AGCM.rc
fi

if [[ $NRL == TRUE ]]
then
   echo "Enabling NRL in AGCM.rc"
   echo "ISOLVAR: 2" >> AGCM.rc
   echo "USE_NRLSSI2: .TRUE." >> AGCM.rc
   echo "SOLAR_CYCLE_FILE_NAME: ExtData/g5gcm/solar/NRLSSI2.v2017.txt" >> AGCM.rc
fi

if [[ $STOCH == TRUE ]]
then
   echo "Enabling Stoch Physics in AGCM.rc"
   echo "# Flags for SPPT scheme" >> AGCM.rc
   echo "SPPT:1" >> AGCM.rc
   echo "SKEB:0" >> AGCM.rc
   echo "STOCH_shutoff:0 " >> AGCM.rc
fi

if [[ $NXY == TRUE ]]
then
   $SED -e "/^ *NX:/ s/\([0-9]\+\)/${NEWNX}/" AGCM.rc
   $SED -e "/^ *NY:/ s/\([0-9]\+\)/${NEWNY}/" AGCM.rc
fi

if [[ $JCAP == TRUE ]]
then
   echo "Setting MKIAU_JCAP to $JCAP_VALUE in AGCM.rc"
   echo "MKIAU_JCAP: $JCAP_VALUE" >> AGCM.rc
fi

print_changes "AGCM.rc"

# ---------------------
# regress/gcm_regress.j
# ---------------------

#if [[ $REPLAY == TRUE ]]
#then
   #echo "Altering regression test for replay"
   #$SED -e "/test_duration = 21/ s/21/18/" \
        #-e "/test_duration = 03/ s/03/06/" regress/gcm_regress.j
#fi

print_changes "regress/gcm_regress.j"

# -----------
# cap_restart
# -----------

if [[ "$DOING_GOCART" == "GOCART" ]]
then
   #USING_NR=$(grep DU_OPTICS AGCM.rc | grep NR)
   USING_OPS=$(grep 'setenv EMISSIONS' gcm_run.j | grep NR)
   if [[ $? -eq 0 ]]
   then
      echo "     You seem to be using the Nature Run GOCART."
      echo "     Setting cap_restart to be in 2005"
      echo ""

      restore_save "cap_restart"
      copy_save "cap_restart"

      $SED -e "/^2000/ s/2000/2005/" cap_restart

      print_changes "cap_restart"
   fi

   #USING_OPS=$(grep DU_OPTICS AGCM.rc | grep g5chem)
   USING_OPS=$(grep 'setenv EMISSIONS' gcm_run.j | grep g5chem)
   if [[ $? -eq 0 ]]
   then
      echo "     You seem to be using the OPS GOCART."
      echo "     Setting cap_restart to be in 2015"
      echo ""

      restore_save "cap_restart"
      copy_save "cap_restart"

      $SED -e "/^2000/ s/2000/2015/" cap_restart

      print_changes "cap_restart"
   fi
fi

if [[ "$OCEAN_RES" == "Ostia-CS" ]]
then
   echo "     You seem to be using the Ostia-CS Ocean."
   echo "     Setting cap_restart to be in 2015"
   echo ""

   restore_save "cap_restart"
   copy_save "cap_restart"

   $SED -e "/^2000/ s/2000/2015/" cap_restart

   print_changes "cap_restart"
fi

if [[ $GAAS == TRUE ]]
then
   echo "     You seem to be using the GAAS."
   echo "     Setting cap_restart to be in 2015"
   echo ""

   restore_save "cap_restart"
   copy_save "cap_restart"

   $SED -e "/^2000/ s/2000/2015/" cap_restart

   print_changes "cap_restart"
fi

if [[ ! -z $NEWBCSDIR ]]
then
   echo "     You seem to be in DYAMOND BCS"
   echo "     Setting cap_restart to be in 2016"
   echo ""

   restore_save "cap_restart"
   copy_save "cap_restart"

   $SED -e "/^2000/ s/2000/2016/" cap_restart

   print_changes "cap_restart"
fi

if [[ $OPSB == TRUE || $OPSBIO == TRUE ]]
then
   echo "     You seem to be using OPSB."
   echo "     Setting cap_restart to be 20181014"
   echo ""

   restore_save "cap_restart"
   copy_save "cap_restart"

   $SED -e "/^20000414/ s/20000414/20181014/" cap_restart

   print_changes "cap_restart"
fi


# ------------------------
# RC/GEOS_ChemGridComp.rc
# ------------------------

#if [[ $TINY == TRUE ]]
#then

   ## Before we turned of GOCART.data. We now provide a
   ## tiny version of it in Tiny-BCs

   ##echo "Turning off GOCART.data and TR in GEOS_ChemGridComp"
   ##$SED -e "/ENABLE_GOCART_DATA:/ s/TRUE/FALSE/" \
        ##-e "/ENABLE_TR:/ s/TRUE/FALSE/" RC/GEOS_ChemGridComp.rc

   ##echo "Turning off TR in GEOS_ChemGridComp"
   ##$SED -e "/ENABLE_TR:/ s/TRUE/FALSE/" RC/GEOS_ChemGridComp.rc
   
   ##print_changes "RC/GEOS_ChemGridComp.rc"
#fi

if [[ $HEMCO == TRUE ]]
then

   echo "Turning on HEMCO in GEOS_ChemGridComp"
   $SED -e "/ENABLE_HEMCO:/ s/FALSE/TRUE/" RC/GEOS_ChemGridComp.rc
   
   print_changes "RC/GEOS_ChemGridComp.rc"
fi

#if [[ $OPSB == TRUE || $OPSBIO == TRUE || $GAAS == TRUE ]]
if [[ $GAAS == TRUE ]]
then

   echo "Turning on GAAS in GEOS_ChemGridComp"
   $SED -e "/ENABLE_GAAS:/ s/FALSE/TRUE/" RC/GEOS_ChemGridComp.rc
   
   print_changes "RC/GEOS_ChemGridComp.rc"
fi

# -------------------
# RC/GAAS_GridComp.rc
# -------------------

#if [[ $OPSB == TRUE || $OPSBIO == TRUE || $GAAS == TRUE ]]
if [[ $GAAS == TRUE ]]
then

   $SED -e "/^aod_ana_filename:/ s#aod_a.sfc.%y4%m2%d2_%h200z.nc4#$UTIL_DIR/GAAS_Input/d5124_m2_jan10.aod_a.sfc.%y4%m2%d2_%h200z.nc4#" \
        -e "/^aod_bkg_filename:/ s#aod_f.sfc.%y4%m2%d2_%h200z.nc4#$UTIL_DIR/GAAS_Input/d5124_m2_jan10.aod_f.sfc.%y4%m2%d2_%h200z.nc4#" \
        -e "/^aod_avk_filename:/ s#aod_k.sfc.%y4%m2%d2_%h200z.nc4#$UTIL_DIR/GAAS_Input/d5124_m2_jan10.aod_k.sfc.%y4%m2%d2_%h200z.nc4#" RC/GAAS_GridComp.rc
   
   print_changes "RC/GAAS_GridComp.rc"
fi

# -------------
# src directory
# -------------

if [[ -d src ]]
then
   echo "src directory found. tarring to save inodes"
   $TAREXEC cf src.tar src
   if [[ -x $PBZIP2 ]]
   then
      echo "pbzip2 found: $PBZIP2, compressing"
      $PBZIP2 -l src.tar
   fi
   echo "removing src directory"
   rm -rf src
fi
