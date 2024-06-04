#! /bin/bash 
#--------------------------------------------------------------------#
#  Sistema de Previsão por Conjunto Global - GDAD/CPTEC/INPE - 2021  #
#--------------------------------------------------------------------#
#BOP
#
# !DESCRIPTION:
# Script para a recomposição dos coeficientes espectrais para ponto de
# grade das previsões do Sistema de Previsão por Conjunto Global 
# (SPCON) do CPTEC.
#
# !INTERFACE:
#      ./run_recfct.sh <opcao1> <opcao2> <opcao3> 
#
# !INPUT PARAMETERS:
#  Opcoes..: <opcao1> resolucao -> resolução espectral do modelo
#                                
#            <opcao2> membro    -> membro controle ou tamanho do
#                                  conjunto
#
#            <opcao3> data      -> data da análise corrente (a partir
#                                  da qual as previsões foram feitas)
#            
#  Uso/Exemplos: ./run_recfct.sh TQ0126L028 CTR 2012123118
#                (recompõe os coeficientes espectrais das previsões
#                feitas a partir da análise controle das 2012123118
#                na resolução TQ0126L028)
#                ./run_recfct.sh TQ0126L028 7 2012123118
#                (recompõe os coeficientes espectrais do conjunto de 7
#                membros das previsões feitas a partir da análise
#                das 2012123118 na resolução TQ0126L028)
# 
# !REVISION HISTORY:
#
# XX Julho de 2017   - C. F. Bastarz - Versão inicial.  
# 16 Agosto de 2017  - C. F. Bastarz - Inclusão comentários.
# 31 Janeiro de 2018 - C. F. Bastarz - Ajustados os prefixos NMC e CTR
# 17 Junho de 2021   - C. F. Bastarz - Ajustes no nome do script de submissão.
# 18 Junho de 2021   - C. F. Bastarz - Revisão geral.
# 26 Outubro de 2022 - C. F. Bastarz - Inclusão de diretivas do SLURM.
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
# Menu de opções/ajuda
#

if [ "${1}" = "help" -o -z "${1}" ]
then
  cat < ${0} | sed -n '/^#BOP/,/^#EOP/p'
  exit 0
fi

#export FILEENV=$(find ./ -name EnvironmentalVariablesMCGA -print)
export FILEENV=$(find ${PWD} -name EnvironmentalVariablesMCGA -print)
#export PATHENV=$(dirname ${FILEENV})
#export PATHBASE=$(cd ${PATHENV}; cd ; pwd)

. ${FILEENV} ${1} ${2}

cd ${HOME_suite}/run

#
# Verificação dos argumentos de entrada
#

if [ -z "${1}" ]
then
  echo "TRCLV está faltando"
  exit 1
else
  TRCLV=${1}
fi

if [ -z "${2}" ]
then
  echo "PREFIC esta faltando"
  exit 1
else
  if ! [[ "${2}" =~ ^[0-9]+$ ]]
  then
    PREFIC=${2}
    TYPES=FCT${PREFIC}
  else
    PREFIC=R
    NMEM=${2}
    TYPES=FCT${PREFIC}PT
  fi
fi

if [ -z "${3}" ]
then
  echo "LABELI esta faltando"
  exit 1
else
  LABELI=${3}
fi

TRC=$(echo ${TRCLV} | cut -c 1-6 | tr -d "TQ0")
LV=$(echo ${TRCLV} | cut -c 7-11 | tr -d "L0")

#
# Variáveis utilizadas no script de submissão
#

HSTMAQ=$(hostname)
RUNTM=$(date +'%y')$(date +'%m')$(date +'%d')$(date +'%H:%M')
EXT=out

mkdir -p ${DK_suite}/recfct/output

#
# Opções específicas para o conjunto de membros ou apenas o controle
#

if [ ${PREFIC} == NMC -o ${PREFIC} == CTR ]
then
  export MODELDATAOUT="cd ${DK_suite}/model/dataout/${TRCLV}/${LABELI}/${PREFIC}/"
  export ENSTYPE="export TYPES=${TYPES}"
else  
  if [ $(echo "$QSUB" | grep qsub) ]
  then
    export PBSDIRECTIVE="#PBS -J 1-${NMEM}"
    export DEFINEMEM="export MEM=\$(printf %02g \${PBS_ARRAY_INDEX})"
  else
    export PBSDIRECTIVE="#SBATCH --array=1-${NMEM}"
    export DEFINEMEM="export MEM=\$(printf %02g \${SLURM_ARRAY_TASK_ID})"
  fi
  export MODELDATAOUT="cd ${DK_suite}/model/dataout/${TRCLV}/${LABELI}/\${MEM}${PREFIC}/"
  export ENSTYPE="export TYPES=FCT\${MEM}${PREFIC}"
fi

RUNTM=$(date +"%s")

#
# Script de submissão
#

SCRIPTSFILE=setrecfct${TYPES}.${TRCLV}.${LABELI}${LABELF}.${MAQUI}

if [ $(echo "$QSUB" | grep qsub) ]
then
  SCRIPTHEADER="
#PBS -o ${DK_suite}/recfct/output/${SCRIPTSFILE}.${RUNTM}.out
#PBS -e ${DK_suite}/recfct/output/${SCRIPTSFILE}.${RUNTM}.err
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=1
#PBS -A CPTEC
#PBS -V
#PBS -S /bin/bash
#PBS -N RECFCT
#PBS -q ${AUX_QUEUE}
${PBSDIRECTIVE}
"
  SCRIPTRUNCMD="aprun -n 1 -N 1 -d 1 ${DK_suite}/recfct/bin/\${TRCLV}/recfct.\${TRCLV} < ${DK_suite}/recfct/datain/recfct\${TYPES}.nml > ${DK_suite}/recfct/output/recfct\${TYPES}.out.\${LABELI}\${LABELF}.\${HOUR}.\${TRCLV}"
  SCRIPTRUNJOB="qsub -W block=true "
else
  SCRIPTHEADER="
#SBATCH --output=${DK_suite}/recfct/output/${SCRIPTSFILE}.${RUNTM}.out
#SBATCH --error=${DK_suite}/recfct/output/${SCRIPTSFILE}.${RUNTM}.err
#SBATCH --time=${AUX_WALLTIME}
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --job-name=RECFCT
#SBATCH --partition=${AUX_QUEUE}
${PBSDIRECTIVE}
"
  if [ $USE_SINGULARITY == true ]
  then          
    SCRIPTRUNCMD="module load singularity ; singularity exec -e --bind ${WORKBIND}:${WORKBIND} ${SIFIMAGE} mpirun -np 1 ${SIFOENSMB09BIN}/recfct/bin/\${TRCLV}/recfct.\${TRCLV} < ${DK_suite}/recfct/datain/recfct\${TYPES}.nml > ${DK_suite}/recfct/output/recfct\${TYPES}.out.\${LABELI}\${LABELF}.\${HOUR}.\${TRCLV}"
  else  
    SCRIPTRUNCMD="mpirun -np 1 ${DK_suite}/recfct/bin/\${TRCLV}/recfct.\${TRCLV} < ${DK_suite}/recfct/datain/recfct\${TYPES}.nml > ${DK_suite}/recfct/output/recfct\${TYPES}.out.\${LABELI}\${LABELF}.\${HOUR}.\${TRCLV}"
  fi  
  if [ ! -z ${job_model_id} ]
  then
    SCRIPTRUNJOB="sbatch --dependency=afterok:${job_model_id}"
  else
    SCRIPTRUNJOB="sbatch "
  fi
fi

monitor=${DK_suite}/recfct/output/monitor_${PREFIC}.t
if [ -e ${monitor} ]; then rm ${monitor}; fi

cat <<EOT0 > ${HOME_suite}/run/${SCRIPTSFILE}
#! /bin/bash -x
${SCRIPTHEADER}

export PBS_SERVER=${pbs_server2}

export TRCLV=${TRCLV}

${DEFINEMEM}

${MODELDATAOUT}

${ENSTYPE}

mkdir -p ${DK_suite}/recfct/datain/

for LABELF in \$(ls G\${TYPES}${LABELI}* | cut -c 18-27)
do 

  #
  # Set date (year,month,day) and hour (hour:minute) 
  #
  # DATE=yyyymmdd
  # HOUR=hh:mn
  #
  
  export DATE=\$(date +'%Y')\$(date +'%m')\$(date +'%d')
  export HOUR=\$(date +'%H:%M')
  echo "Date: "\$DATE
  echo "Hour: "\$HOUR
  
  #
  # LABELI = yyyymmddhh
  # LABELI = input file start label
  #
  
  export LABELI=${LABELI}
  
  #
  # Prefix names for the FORTRAN files
  #
  # NAMEL - List file name prefix
  # NAMES - Input spectral file name prefix
  # NAMER - Output gridded file name prefix
  #
  # Suffix names for the FORTRAN files
  #
  # EXTL - List file name suffix
  # ERSi - Input spectral file name suffix
  # ERRi - Output gridded file name suffix
  #
  
  export NAMEL=G\${TYPES}
  export NAMES=G\${TYPES}
  export NAMER=G\${TYPES}
  
  if [ \${TYPES} = ANLAVN ] 
  then
    export EXTL=S.unf
    export ERS1=S.unf
    export ERR1=R.unf
  else
    export EXTL=F.fct
    export ERS1=F.fct
    export ERR1=R.fct
  fi
  
  #
  # Now, build the necessary NAMELIST input:
  #
  
  GNAMEL=\${NAMEL}\${LABELI}\${LABELF}\${EXTL}.\${TRCLV}
  echo \${GNAMEL}
  echo ${DK_suite}/recfct/datain/\${GNAMEL}
  
cat <<EOT2 > ${DK_suite}/recfct/datain/\${GNAMEL}
\${NAMES}\${LABELI}\${LABELF}\${ERS1}.\${TRCLV}
\${NAMER}\${LABELI}\${LABELF}\${ERR1}.\${TRCLV}
EOT2

cat <<EOT3 > ${DK_suite}/recfct/datain/recfct\${TYPES}.nml
 &DATAIN
  LDIM=1
  DIRL='${DK_suite}/recfct/datain/ '
  DIRS='${DK_suite}/model/dataout/\${TRCLV}/\${LABELI}/\${TYPES:3}/  '
  DIRR='${DK_suite}/recfct/dataout/\${TRCLV}/\${LABELI}/ '
  GNAMEL='\${GNAMEL} '
 &END
EOT3

  mkdir -p ${DK_suite}/recfct/dataout/\${TRCLV}/\${LABELI}/

  #
  # Run Decomposition
  #
  
  cd ${HOME_suite}/recfct/bin/\${TRCLV}
  
  ${SCRIPTRUNCMD} 
done

touch ${monitor}
EOT0

#
# Submete o script e aguarda o fim da execução
#

export PBS_SERVER=${pbs_server2}

chmod +x ${HOME_suite}/run/${SCRIPTSFILE}

job_recfct=$(${SCRIPTRUNJOB} ${HOME_suite}/run/${SCRIPTSFILE})
export job_recfct_id=$(echo ${job_recfct} | awk -F " " '{print $4}')
echo "recfct ${job_recfct_id}"

until [ -e ${monitor} ]; do sleep 1s; done

#exit 0
