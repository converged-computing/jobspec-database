#! /bin/bash 
#--------------------------------------------------------------------#
#  Sistema de Previsão por Conjunto Global - GDAD/CPTEC/INPE - 2021  #
#--------------------------------------------------------------------#
#BOP
#
# !DESCRIPTION:
# Script para o cálculo dos clusters do conjunto de previsões em 
# ponto de grade do Sistema de Previsão por Conjunto Global (SPCON) 
# do CPTEC.
#
# !INTERFACE:
#      ./run_cluster.sh <opcao1> <opcao2> <opcao3> <opcao4> <opcao5> 
#
# !INPUT PARAMETERS:
#  Opcoes..: <opcao1> resolucao -> resolução espectral do modelo
#
#            <opcao2> data      -> data da análise corrente (a partir
#                                  da qual as previsões foram feitas)
#
#            <opcao3> dias      -> dias de previsões que serão consideradas
#                                  a partir da data da análise
#
#            <opcao4> prefixo   -> prefixo que identifica o tipo de análise
#
#            <opcao5> membro    -> tamanho do conjunto
#            
#  Uso/Exemplos: ./run_cluster.sh TQ0126L028 2020031300 15 NMC 7
#                (calcula a média do conjunto das previsões entre
#                2020031300 e 2020032800 considerando as 7 perturbações 
#                N e Pmembros na resolução TQ0126L028) 
#
# !REVISION HISTORY:
#
# 25 Maio de 2020      - C. F. Bastarz - Versão inicial.  
# 18 Junho de 2021     - C. F. Bastarz - Revisão geral.
# 01 Novembro de 2022  - C. F. Bastarz - Inclusão de diretivas do SLURM.
# 06 Fevereiro de 2023 - C. F. Bastarz - Adaptações para a Egeon.
#
# !REMARKS:
#
# !BUGS:
#
#EOP  
#--------------------------------------------------------------------#
#BOC

# Descomentar para debugar
#set -o xtrace

#
# Menu de ajuda
#

if [ "${1}" = "help" -o -z "${1}" ]
then
  cat < ${0} | sed -n '/^#BOP/,/^#EOP/p'
  exit 0
fi

#
# Argumentos da linha de comando
#

if [ -z ${1} ]
then
  echo "RES esta faltando"
  exit 1
else
  export RES=${1}
fi

if [ -z ${2} ]
then
  echo "LABELI esta faltando"
  exit 1
else
  export LABELI=${2}
fi

if [ -z ${3} ]
then
  echo "NFCTDY esta faltando"
  exit 1
else
  export NFCTDY=${3}
fi

if [ -z ${4} ]
then
  echo "PREFX esta faltando"
  exit 1
else
  export PREFX=${4}
fi

if [ -z ${5} ]
then
  echo "NRNDP esta faltando"
  exit 1
else
  export NRNDP=${5}
fi

export FILEENV=$(find ./ -name EnvironmentalVariablesMCGA -print)
export PATHENV=$(dirname ${FILEENV})
export PATHBASE=$(cd ${PATHENV}; cd ; pwd)

. ${FILEENV} ${RES} ${PREFX}

cd ${HOME_suite}/run

TRC=$(echo ${TRCLV} | cut -c 1-6 | tr -d "TQ0")
LV=$(echo ${TRCLV} | cut -c 7-11 | tr -d "L0")

export RESOL=${TRCLV:0:6}
export NIVEL=${TRCLV:6:4}

#
# Cálculo do tamanho total do conjunto
#

export NMEMBR=$((2*${NRNDP}+1))

export LABELF=$(${inctime} ${LABELI} +${NFCTDY}d %y4%m2%d2%h2)

case ${TRC} in
  021) MR=22  ; IR=64  ; JR=32  ; NPGH=93   ; DT=1800 ;;
  030) MR=31  ; IR=96  ; JR=48  ; NPGH=140  ; DT=1800 ;;
  042) MR=43  ; IR=128 ; JR=64  ; NPGH=187  ; DT=1800 ;;
  047) MR=48  ; IR=144 ; JR=72  ; NPGH=26   ; DT=1200 ;;
  062) MR=63  ; IR=192 ; JR=96  ; NPGH=315  ; DT=1200 ;;
  079) MR=80  ; IR=240 ; JR=120 ; NPGH=26   ; DT=900  ;;
  085) MR=86  ; IR=256 ; JR=128 ; NPGH=26   ; DT=720  ;;
  094) MR=95  ; IR=288 ; JR=144 ; NPGH=591  ; DT=720  ;;
  106) MR=107 ; IR=320 ; JR=160 ; NPGH=711  ; DT=600  ;;
  126) MR=127 ; IR=384 ; JR=192 ; NPGH=284  ; DT=600  ;;
  159) MR=160 ; IR=480 ; JR=240 ; NPGH=1454 ; DT=450  ;;
  170) MR=171 ; IR=512 ; JR=256 ; NPGH=1633 ; DT=450  ;;
  213) MR=214 ; IR=640 ; JR=320 ; NPGH=2466 ; DT=360  ;;
  254) MR=255 ; IR=768 ; JR=384 ; NPGH=3502 ; DT=300  ;;
  319) MR=320 ; IR=960 ; JR=480 ; NPGH=26   ; DT=240  ;;
  *) echo "Wrong request for horizontal resolution: ${TRC}" ; exit 1;
esac

export RUNTM=$(date +'%Y%m%d%T')

export OPERM=${DK_suite}
export ROPERM=${DK_suite}/produtos

# 
# Script de submissão
#

cd ${OPERM}/run

export SCRIPTFILEPATH1=${DK_suite}/run/setcluster.${RESOL}${NIVEL}.${LABELI}.${MAQUI}
export SCRIPTFILEPATH2=${DK_suite}/run/setcluster_figs.${RESOL}${NIVEL}.${LABELI}.${MAQUI}

if [ $(echo "$QSUB" | grep qsub) ]
then
  SCRIPTHEADER1="
#PBS -o ${ROPERM}/cluster/output/cluster.${RUNTM}.out
#PBS -e ${ROPERM}/cluster/output/cluster.${RUNTM}.err
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1
#PBS -A CPTEC
#PBS -V
#PBS -S /bin/bash
#PBS -N CLUSTER
#PBS -q ${AUX_QUEUE}
"
  SCRIPTHEADER2="
#PBS -o ${ROPERM}/cluster/output/cluster_figs.${RUNTM}.out
#PBS -e ${ROPERM}/cluster/output/cluster_figs.${RUNTM}.err
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=1
#PBS -A CPTEC
#PBS -V
#PBS -S /bin/bash
#PBS -N CLUSTERFIGS
#PBS -q ${AUX_QUEUE}
"
  SCRIPTRUNCMD="aprun -n 1 -N 1 -d 1 \${ROPERMOD}/cluster/bin/cluster.x ${LABELI} ${LABELF} > \${ROPERMOD}/cluster/output/cluster.${RUNTM}.log"
  SCRIPTRUNJOB="qsub -W block=true "
else
  SCRIPTHEADER1="
#SBATCH --output=${ROPERM}/cluster/output/cluster.${RUNTM}.out
#SBATCH --error=${ROPERM}/cluster/output/cluster.${RUNTM}.err
#SBATCH --time=00:10:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --job-name=CLUSTER
#SBATCH --partition=${AUX_QUEUE}
"
  SCRIPTHEADER2="
#SBATCH --output=${ROPERM}/cluster/output/cluster_figs.${RUNTM}.out
#SBATCH --error=${ROPERM}/cluster/output/cluster_figs.${RUNTM}.err
#SBATCH --time=00:30:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --job-name=CLUSTERFIGS
#SBATCH --partition=${AUX_QUEUE}
"
  if [ $USE_SINGULARITY == true ]
  then
    SCRIPTRUNCMD="module load singularity ; singularity exec -e --bind ${WORKBIND}:${WORKBIND} ${SIFIMAGE} mpirun -np 1 ${SIFOENSMB09BIN}/produtos/cluster/bin/cluster.x ${LABELI} ${LABELF} > \${ROPERMOD}/cluster/output/cluster.${RUNTM}.log"
  else
    SCRIPTRUNCMD="mpirun -np 1 \${ROPERMOD}/cluster/bin/cluster.x ${LABELI} ${LABELF} /ensmed/output/ensmed.${RUNTM}.log"
  fi
  SCRIPTRUNJOB="sbatch "
fi

if [ -e ${ROPERM}/cluster/bin/cluster-${LABELI}.ok ]; then rm ${ROPERM}/cluster/bin/cluster-${LABELI}.ok; fi

if [ -e ${ROPERM}/cluster/bin/cluster_figs-${LABELI}.ok ]; then rm ${ROPERM}/cluster/bin/cluster_figs-${LABELI}.ok; fi

cat <<EOT0 > ${SCRIPTFILEPATH1}
#! /bin/bash -x
${SCRIPTHEADER1}

export DATE=$(date +'%Y%m%d')
export HOUR=$(date +'%T')

# OPERMOD is the directory for sources, scripts and printouts files.
# SOPERMOD is the directory for input and output data and bin files.
# ROPERMOD is the directory for big selected output files.

export OPERMOD=${OPERM}
export ROPERMOD=${ROPERM}
export LEV=${NIVEL}
export TRUNC=${RESOL}
export MACH=${MAQUI}
export EXTS=S.unf

# Parameter to be read by cluster.f90 : namelist file
# IMAX      : ( INTEGER ) number of points in zonal direction
# JMAX      : ( INTEGER ) number of points in merdional direction
# NMEMBERS  : ( INTEGER ) number of members of the ensemble
# NFCTDY    : ( INTEGER ) number of forecast days
# GRPETA    : ( INTEGER ) number of outputs clusters for eta ensemble
# FREQCALC  : ( INTEGER ) interval in hours for computing clusters
# LONW      : ( REAL    ) western longitude for region used to evaluate the clusters 
# LONE      : ( REAL    ) eastern longitude for region used to evaluate the clusters 
# LATS      : ( REAL    ) southern latitude for region used to evaluate the clusters 
# LATN      : ( REAL    ) northern latitude for region used to evaluate the clusters 
# DATALSTDIR: ( CHAR    ) input directory (ensemble members)
# DATARMSDIR: ( CHAR    ) input directory of climatological rms of GCM-CPTEC
# DATAOUTDIR: ( CHAR    ) output directory of cluster means
# DATACLTDIR: ( CHAR    ) output directory of cluster tables
# RESOL     : ( CHAR    ) horizontal and vertical model resolution
# PREFX     : ( CHAR    ) preffix for input and output files 

mkdir -p \${ROPERMOD}/cluster/dataout/\${TRUNC}\${LEV}/\${LABELI}/clusters/
mkdir -p \${ROPERMOD}/cluster/rmsclim

cat <<EOF0 > \${ROPERMOD}/cluster/bin/clustersetup.${LABELI}.nml
IMAX      :   ${IR}
JMAX      :   ${JR}
NMEMBERS  :   ${NMEMBR}
NFCTDY    :   ${NFCTDY}
GRPETA    :    4
FREQCALC  :    6
LONW      :   -101.25 
LONE      :    -11.25 
LATS      :    -60.00
LATN      :     15.00
DATALSTDIR:   \${OPERMOD}/pos/dataout/\${TRUNC}\${LEV}/\${LABELI}/
DATARMSDIR:   \${ROPERMOD}/cluster/rmsclim/
DATAOUTDIR:   \${ROPERMOD}/cluster/dataout/\${TRUNC}\${LEV}/\${LABELI}/
DATACLTDIR:   \${ROPERMOD}/cluster/dataout/\${TRUNC}\${LEV}/\${LABELI}/clusters/
RESOL     :   \${TRUNC}\${LEV}
PREFX     :   ${PREFX}
EOF0

cd \${ROPERMOD}/cluster/bin

${SCRIPTRUNCMD}

echo "" > \${ROPERMOD}/cluster/bin/cluster-${LABELI}.ok
EOT0

#
# Figuras
#

cat <<EOT1 > ${SCRIPTFILEPATH2}
#! /bin/bash -x
${SCRIPTHEADER2}

export DATE=$(date +'%Y%m%d')
export HOUR=$(date +'%T')

# OPERMOD is the directory for sources, scripts and printouts files.
# SOPERMOD is the directory for input and output data and bin files.
# ROPERMOD is the directory for big selected output files.

export OPERMOD=${OPERM}
export ROPERMOD=${ROPERM}
export LEV=${NIVEL}
export TRUNC=${RESOL}
export MACH=${MAQUI}
export EXTS=S.unf
                                                                                      
DIRSCR=${ROPERM}/cluster/scripts
DIRGIF=${ROPERM}/cluster/gif/${LABELI}
DIRCTL=${OPERM}/pos/dataout/${RES}/${LABELI}
DIRCLT=${ROPERM}/cluster/dataout/${RES}/${LABELI}/clusters

if [ ! -d \${DIRGIF} ]; then mkdir -p \${DIRGIF}; fi
if [ ! -d \${DIRCTL} ]; then mkdir -p \${DIRCTL}; fi

#
# Lista de arquivos descritores (ctl) a serem abertos
#

cd \${DIRSCR}

NPERT=1
while [ \${NPERT} -le ${NRNDP} ]
do
  if [ \${NPERT} -lt 10 ]; then NPERT='0'\${NPERT}; fi

  rm -f filefct\${NPERT}P${LABELI}.${TRC}
  rm -f filefct\${NPERT}N${LABELI}.${TRC}

  let NPERT=NPERT+1
done

rm -f filefct${PREFX}${LABELI}.${TRC}
rm -f fileclt${LABELI}.${TRC}

let NHOURS=24*NFCTDY

NCTLS=0
TIM=0

while [ \${TIM} -le \${NHOURS} ]
do
  LABELF=\$(${inctime} ${LABELI} +\${TIM}hr %y4%m2%d2%h2)
  echo 'LABELF='${LABELF}

  if [ \${TIM} -eq 0 ]; then TYPE='P.icn'; else TYPE='P.fct'; fi

  NPERT=1
  while [ \${NPERT} -le ${NRNDP} ]
  do
    if [ \${NPERT} -lt 10 ]; then  NPERT='0'\${NPERT}; fi

    if [ -s \${DIRCTL}/GPOS\${NPERT}P${LABELI}\${LABELF}\${TYPE}.${RES}.ctl ]
    then

cat <<EOF0 >> filefct\${NPERT}P${LABELI}.${TRC}
\${DIRCTL}/GPOS\${NPERT}P${LABELI}\${LABELF}\${TYPE}.${RES}.ctl
EOF0

    else
       echo "\${DIRCTL}/GPOS\${NPERT}P${LABELI}\${LABELF}\${TYPE}.${RES}.ctl nao existe"
       exit 2
    fi

    if [ -s \${DIRCTL}/GPOS\${NPERT}N${LABELI}\${LABELF}\${TYPE}.${RES}.ctl ]
    then

cat <<EOF1 >> filefct\${NPERT}N${LABELI}.${TRC}
\${DIRCTL}/GPOS\${NPERT}N${LABELI}\${LABELF}\${TYPE}.${RES}.ctl
EOF1

    else
      echo "\${DIRCTL}/GPOS\${NPERT}N${LABELI}\${LABELF}\${TYPE}.${RES}.ctl nao existe"
      exit 2
    fi

    let NPERT=NPERT+1
  done

  if [ -s \${DIRCTL}/GPOS\${PREFX}${LABELI}\${LABELF}\${TYPE}.${RES}.ctl ]
  then

cat <<EOF2 >> filefct\${PREFX}${LABELI}.${TRC}
\${DIRCTL}/GPOS\${PREFX}${LABELI}\${LABELF}\${TYPE}.${RES}.ctl
EOF2

  else
    echo "\${DIRCTL}/GPOS\${PREFX}${LABELI}\${LABELF}\${TYPE}.${RES}.ctl nao existe"
    exit 2
  fi

  if [ -s \${DIRCLT}/clusters${LABELI}\${LABELF}.${RES} ]
  then

cat <<EOF3 >> fileclt${LABELI}.${TRC}
\${DIRCLT}/clusters${LABELI}\${LABELF}.${RES}
EOF3

  else
    echo "\${DIRCLT}/clusters${LABELI}\${LABELF}.${RES} nao existe"
    exit 2
  fi

  let NCTLS=NCTLS+1
  let TIM=TIM+6
done

echo "NCTLS="\${NCTLS}

${DIRGRADS}/grads -bp << EOF4
run plot_temp_zgeo.gs
${TRC} ${LABELI} ${NMEMBR} \${NCTLS} ${RES} ${PREFX} \${DIRGIF} ${convert}
EOF4

${DIRGRADS}/grads -bp << EOF5
run plot_prec_psnm_wind.gs
${TRC} ${LABELI} ${NMEMBR} \${NCTLS} ${RES} ${PREFX} \${DIRGIF} ${convert}
EOF5

${DIRGRADS}/grads -bp << EOF6
run plot_tems.gs
${TRC} ${LABELI} ${NMEMBR} \${NCTLS} ${RES} ${PREFX} \${DIRGIF} ${convert}
EOF6

#
# Remove os arquivos temporários
#

NPERT=1
while [ \${NPERT} -le ${NRNDP} ]
do
  if [ \${NPERT} -lt 10 ]; then NPERT='0'\${NPERT}; fi

  rm -f filefct\${NPERT}P${LABELI}.${TRC}
  rm -f filefct\${NPERT}N${LABELI}.${TRC}
  let NPERT=NPERT+1
done

rm -f filefct${PREFX}${LABELI}.${TRC}
rm -f fileclt${LABELI}.${TRC}

echo "" > \${ROPERMOD}/cluster/bin/cluster_figs-${LABELI}.ok
EOT1

#
# Submissão
# 

export PBS_SERVER=${pbs_server2}

chmod +x ${SCRIPTFILEPATH1}

${SCRIPTRUNJOB} ${SCRIPTFILEPATH1}

until [ -e "${ROPERM}/cluster/bin/cluster-${LABELI}.ok" ]; do sleep 1s; done

chmod +x ${SCRIPTFILEPATH2}

${SCRIPTRUNJOB} ${SCRIPTFILEPATH2}

until [ -e "${ROPERM}/cluster/bin/cluster_figs-${LABELI}.ok" ]; do sleep 1s; done

if [ ${SEND_TO_FTP} == true ]
then
  cd ${ROPERM}/cluster/gif/${LABELI}/
  ls *.png >  list.txt
  rsync -arv * ${FTP_ADDRESS}/cluster/${LABELI}/
fi

#exit 0
