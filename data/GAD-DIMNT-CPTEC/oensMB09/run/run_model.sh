#! /bin/bash 
#--------------------------------------------------------------------#
#  Sistema de Previsão por Conjunto Global - GDAD/CPTEC/INPE - 2021  #
#--------------------------------------------------------------------#
#BOP
#
# !DESCRIPTION:
# Script para submeter o modelo atmosférico para a integração das
# análises do Sistema de Previsão por Conjunto Global (SPCON) do CPTEC.
#
# !INTERFACE:
#      ./run_model.sh <opcao1> <opcao2> <opcao3> <opcao4> <opcao5>
#                      <opcao6> <opcao7> <opcao8> <opcao9> <opcao10>
#
# !INPUT PARAMETERS:
#  Opcoes..: <opcao1> num_proc  -> número de processadores
#            
#            <opcao2> num_nos   -> número de cores por nó
#            
#            <opcao3> num_omp   -> número de processos omp por
#                                  processo mpi
#
#            <opcao4> resolucao -> resolução espectral do modelo
#                                
#            <opcao5> pref_topo -> prefixo que identifica o tipo de
#                                  análise de acordo com topografia
#
#            <opcao6> datai     -> data da análise corrente 
#
#            <opcao7> dataf     -> data da previsão final
#
#            <opcao8> prefixo   -> prefixo que identifica o tipo de análise 
#            
#            <opcao9> init      -> tipo de inicialização 
#
#            <opcao10> n_mem b  -> tamanho do conjunto de perturbações
#            
#  Uso/Exemplos: 
# 
#  Membro controle:
# - previsões a partir da análise controle (prefixo NMC - análises NCEP ou ECMWF): 
# ./run_model.sh 48 24 1 TQ0126L028 SMT 2013010100 2013010300 NMC 2 1
# - previsões a partir da análise controle (prefixo CTR - análises NCEP):
# ./run_model.sh 48 24 1 TQ0126L028 SMT 2013010100 2013011600 CTR 2 1
# - previsões a partir da análise controle (prefixo EIT - análises ECMWF de 1,5 graus): 
# ./run_model.sh 48 24 1 TQ0126L028 SMT 2013010100 2013011600 EIT 2 1
#  - previsoes a partir das analises do ECMWF (prefixo EIH - análises ECMWF de 0,75 graus)
# ./run_model.sh 48 24 1 TQ0126L028 EIH 2013010100 2013011600 EIH 2 1
# 
#  Demais membros:
#  - previsões a partir das analises perturbadas randomicamente:
# ./run_model.sh 48 24 1 TQ0126L028 SMT 2013010100 2013010300 RDP 2 7
#  - previses a partir das analises perturbadas por EOF (subtraidas):
# ./run_model.sh 48 24 1 TQ0126L028 SMT 2013010100 2013010300 NPT 2 7
#  - previsões a partir das analises perturbadas por EOF (somadas):
# ./run_model.sh 48 24 1 TQ0126L028 SMT 2013010100 2013010300 PPT 2 7
#
# !REVISION HISTORY:
# 
# XX Julho de 2017   - C. F. Bastarz - Versão inicial.  
# 16 Agosto de 2017  - C. F. Bastarz - Inclusão comentários.
# 17 Agosto de 2017  - C. F. Bastarz - Inclusão da opção <dataf>
# 18 Agosto de 2017  - C. F. Bastarz - Modificação nos argumentos de entrada.
# 22 Agosto de 2017  - C. F. Bastarz - Inclusão do sleep 10s no final do script
#                                     de submissão para aguardar o I/O do BAM
# 26 Outubro de 2017 - C. F. Bastarz - Inclusão dos prefixos das análises do ECMWF (EIT/EIH)
# 25 Janeiro de 2018 - C. F. Bastarz - Ajuste dos prefixos NMC (controle 48h) e CTR (controle 120h)
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

cria_namelist() {

sed -e "s;#TRUNC#;${1};g" \
    -e "s;#NLEV#;${2};g" \
    -e "s;#DELT#;${3};g" \
    -e "s;#LABELI#;${4:8:2},${4:6:2},${4:4:2},${4:0:4};g" \
    -e "s;#LABELW#;${5:8:2},${5:6:2},${5:4:2},${5:0:4};g" \
    -e "s;#LABELF#;${6:8:2},${6:6:2},${6:4:2},${6:0:4};g" \
    -e "s;#DHFCT#;${7};g" \
    -e "s;#DHRES#;${8};g" \
    -e "s;#GENRES#;${9};g" \
    -e "s;#PREFIX#;${10};g" \
    -e "s;CPT;${11};g" \
    -e "s;#NMSST#;${12};g" \
    -e "s;#PATHIN#;${13};g" \
    -e "s;#PATHOU#;${14};g" \
    -e "s;#RSTIN#;${15};g" \
    -e "s;#RSTOU#;${16};g" \
    -e "s;#EIGENINIT#;${17};g" \
    -e "s;#MGIVEN#;${18};g" \
    -e "s;#GAUSSGIVEN#;${19};g" \
    -e "s;#INITLZ#;${20};g" \
    -e "s;#NHEXT#;${21};g" \
    -e "s;#DOGRH#;${22};g" \
    ${23}/MODELIN.template > ${24}/MODELIN

echo "Namelist criado em: ${24}/MODELIN"

}

#
# Verificação dos argumentos de entrada
#

if [ -z "${1}" ]
then
  echo "MPPWIDTH esta faltando" 
  exit 3
else
  export MPPWIDTH=${1}  
fi

if [ -z "${2}" ]
then
  echo "MPPNPPN esta faltando" 
  exit 3
else
  export MPPNPPN=${2}  
fi

if [ -z "${3}" ]
then
  echo "MPPDEPTH esta faltando" 
  exit 3
else
  export MPPDEPTH=${3}  
fi

if [ -z "${4}" ]
then
  echo "RESOL esta faltando" 
  exit 3
else
  export RES=${4}  
fi

if [ -z "${5}" ]
then
  echo "PREFIC esta faltando" 
  exit 3
else
  export PREFIC=${5}  
fi

if [ -z "${6}" ]
then
  echo "LABELI esta faltando" 
  exit 3
else
  export LABELI=${6} 
fi

if [ -z "${7}" ]
then
  echo "LABELF esta faltando" 
  exit 3
else
  export LABELF=${7} 
fi

if [ -z "${8}" ]
then
  echo "ANLTYPE esta faltando" 
  exit 3
else
  export ANLTYPE=${8}  
fi

if [ -z "${9}" ]
then
  echo "INITLZ esta faltando" 
  exit 3
else
  export INITLZ=${9}  
fi

if [ -z "${10}" ]
then
  echo "ANLPERT esta faltando" 
else
  export ANLPERT=${10}  
fi

#export FILEENV=$(find ./ -name EnvironmentalVariablesMCGA -print)
export FILEENV=$(find ${PWD} -name EnvironmentalVariablesMCGA -print)
#export PATHENV=$(dirname ${FILEENV})
#export PATHBASE=$(cd ${PATHENV}; cd ; pwd)

. ${FILEENV} ${RES} ${PREFIC}

cd ${HOME_suite}/run

TRC=$(echo ${TRCLV} | cut -c 1-6 | tr -d "TQ0")
LV=$(echo ${TRCLV} | cut -c 7-11 | tr -d "L0")

export RESOL=${TRCLV:0:6}
export NIVEL=${TRCLV:6:4}

#
# Se a previsão for a controle para a perturbação, integra o modelo por apenas 48 horas;
# Se a previsão for a controle final, integra o modelo por 360 horas (15 dias);
# Se a previsão for a partir do conjunto de perturbações por EOF, integra o modelo por 360 horas (15 dias).
#

export LABELW=${LABELF}

if [ ${TRCLV} == "TQ0126L028" ]
then
  export TIMESTEP=600
elif [ ${TRCLV} == "TQ0213L042" ]
then
  export TIMESTEP=360
elif [ ${TRCLV} == "TQ0299L064" ]
then
  export TIMESTEP=200
else
  echo "Erro na resolução ${TRCLV}"
  exit 1
fi

DIRRESOL=$(echo ${TRC} ${LV} | awk '{printf("TQ%4.4dL%3.3d\n",$1,$2)}')
MAQUI=$(hostname -s)

SCRIPTFILEPATH=${HOME_suite}/run/set$(echo "${ANLTYPE}" | awk '{print tolower($0)}')${ANLPERT}modg.${DIRRESOL}.${LABELI}.${MAQUI}
NAMELISTFILEPATH=${HOME_suite}/run

#
# As opções abaixo fazem referência à frequência de saída das previsões (DHFCT) e dos arquivos de restart (DHRES)
# Se ANLTYPE for igual a NMC ou RDP, então as previsões serão referentes à análise controle, com previsões para
# 2 dias e com saídas a cada 3 horas;
# Se ANLTYPE for igual a CTR, NPT ou PPT, então as previsões serão referentes às análises controle e perturbadas
# por EOF (respectivamente), e serão feitas para 15 dias e com saída a cada 3 horas.
#

if [ ${ANLTYPE} == RDP -o ${ANLTYPE} == CTR ]
then
  export DHFCT=3
  export DHRES=24
  export NHEXT=36
  export DOGRH='.FALSE.'
elif [ ${ANLTYPE} == NMC -o ${ANLTYPE} == NPT -o ${ANLTYPE} == PPT -o ${ANLTYPE} == EIT -o ${ANLTYPE} == EIH ]
then
  export DHFCT=6
  export DHRES=24
  export NHEXT=168
  export DOGRH='.TRUE.'
else
  export DHFCT=6
  export DHRES=24
  export NHEXT=36
  export DOGRH='.FALSE.'
fi

export NMSST="sstwkl"

export GENRES='.FALSE.'
export EIGENINIT=".FALSE."
export MGIVEN=".TRUE."      
export GAUSSGIVEN=".TRUE."  

export PATHIN=${DK_suite}/model/datain

#
# Variáveis utilizadas no script de submissão
#

if [ ${ANLTYPE} == CTR -o ${ANLTYPE} == NMC -o ${ANLTYPE} == EIT -o ${ANLTYPE} == EIH ]
then

  EXECFILEPATH=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}

  mkdir -p ${EXECFILEPATH}/setout
  
  if [ ${USE_SINGULARITY} != true ]
  then 
    ln -sf ${DK_suite}/model/exec/ParModel_MPI ${EXECFILEPATH}
  fi

  export RSTIN=${DK_suite}/model/dataout/${TRCLV}/${LABELI}/${ANLTYPE}/RST
  export RSTOU=${DK_suite}/model/dataout/${TRCLV}/${LABELW}/${ANLTYPE}/RST
  export DIRFNAMEOUTPUT=${DK_suite}/model/dataout/${DIRRESOL}/${LABELI}/${ANLTYPE}

  mkdir -p ${DIRFNAMEOUTPUT}

  if [ ${ANLTYPE} == CTR ]
  then
    export PREFIY=CTR
  elif [ ${ANLTYPE} == EIT ]
  then
    export PREFIY=EIT
  elif [ ${ANLTYPE} == EIH ]
  then
    export PREFIY=EIH
  else
    export PREFIY=NMC
  fi
  export PREFIX=${PREFIC}

  cria_namelist ${TRC} ${LV} ${TIMESTEP} ${LABELI} ${LABELW} ${LABELF} ${DHFCT} ${DHRES} ${GENRES} ${PREFIX} ${PREFIY} ${NMSST} ${PATHIN} ${DIRFNAMEOUTPUT} ${RSTIN} ${RSTOU} ${EIGENINIT} ${MGIVEN} ${GAUSSGIVEN} ${INITLZ} ${NHEXT} ${DOGRH} ${NAMELISTFILEPATH} ${EXECFILEPATH}
 
else

  for MEM in $(seq -f %02g 1 ${ANLPERT})
  do

    EXECFILEPATH=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}
    EXECFILEPATHMEM=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/${MEM}${ANLTYPE:0:1}

    #mkdir -p ${EXECFILEPATH}/setout ${EXECFILEPATHMEM}
    mkdir -p ${EXECFILEPATH}/setout ${EXECFILEPATHMEM}/setout
   
    if [ ${USE_SINGULARITY} != true ]
    then 
      ln -sf ${DK_suite}/model/exec/ParModel_MPI ${EXECFILEPATHMEM}
    fi  

    ln -sf ${DK_suite}/model/datain/OZON${PREFIC}${LABELI}S.grd.G00192L028 ${DK_suite}/model/datain/OZON${MEM}${ANLTYPE:0:1}${LABELI}S.grd.G00192L028
    ln -sf ${DK_suite}/model/datain/TRAC${PREFIC}${LABELI}S.grd.G00192L028 ${DK_suite}/model/datain/TRAC${MEM}${ANLTYPE:0:1}${LABELI}S.grd.G00192L028

    export RSTIN=${DK_suite}/model/dataout/${TRCLV}/${LABELI}/${MEM}${ANLTYPE:0:1}/RST
    export RSTOU=${DK_suite}/model/dataout/${TRCLV}/${LABELW}/${MEM}${ANLTYPE:0:1}/RST
    export DIRFNAMEOUTPUT=${DK_suite}/model/dataout/${DIRRESOL}/${LABELI}/${MEM}${ANLTYPE:0:1}

    mkdir -p ${DIRFNAMEOUTPUT}

    export PREFIY=${MEM}${ANLTYPE:0:1}
    export PREFIX=${MEM}${ANLTYPE:0:1}

    cria_namelist ${TRC} ${LV} ${TIMESTEP} ${LABELI} ${LABELW} ${LABELF} ${DHFCT} ${DHRES} ${GENRES} ${PREFIX} ${PREFIY} ${NMSST} ${PATHIN} ${DIRFNAMEOUTPUT} ${RSTIN} ${RSTOU} ${EIGENINIT} ${MGIVEN} ${GAUSSGIVEN} ${INITLZ} ${NHEXT} ${DOGRH} ${NAMELISTFILEPATH} ${EXECFILEPATHMEM}

  done

fi

if [ ${ANLTYPE} != CTR -a ${ANLTYPE} != NMC -a ${ANLTYPE} != EIT -a ${ANLTYPE} != EIH ]
then
  if [ $(echo "$QSUB" | grep qsub) ]
  then
    export PBSOUTFILE="#PBS -o ${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.out"
    export PBSERRFILE="#PBS -e ${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.err"
    export PBSDIRECTIVENAME="#PBS -N BAMENS${ANLTYPE}"
    export PBSDIRECTIVEARRAY="#PBS -J 1-${ANLPERT}"
    export PBSMEM="export MEM=\$(printf %02g \${PBS_ARRAY_INDEX})"
  else
    export PBSOUTFILE="#SBATCH --output=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.out"
    export PBSERRFILE="#SBATCH --error=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.err"
    export PBSDIRECTIVENAME="#SBATCH --job-name=BAMENS${ANLTYPE}"
    export PBSDIRECTIVEARRAY="#SBATCH --array=1-${ANLPERT}"
    export PBSMEM="export MEM=\$(printf %02g \${SLURM_ARRAY_TASK_ID})"
  fi
  export PBSEXECFILEPATH="export EXECFILEPATH=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/\${MEM}${ANLTYPE:0:1}"
else
  if [ $(echo "$QSUB" | grep qsub) ]
  then
    export PBSOUTFILE="#PBS -o ${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.out"
    export PBSERRFILE="#PBS -e ${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.err"
    export PBSDIRECTIVENAME="#PBS -N BAM${ANLTYPE}"
  else
    export PBSOUTFILE="#SBATCH --output=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.out"
    export PBSERRFILE="#SBATCH --error=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/Out.model.${LABELI}.MPI${MPPWIDTH}.err"
    export PBSDIRECTIVENAME="#SBATCH --job-name=BAM${ANLTYPE}"
  fi
  export PBSDIRECTIVEARRAY=""
  export PBSMEM=""
  export PBSEXECFILEPATH="export EXECFILEPATH=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}"
fi

##if [ ${ANLTYPE} != NMC -a ${ANLTYPE} != RDP ]
#if [ ${ANLTYPE} != CTR -a ${ANLTYPE} != RDP ]
#then
  export walltime="04:00:00"
#else
#  export walltime="02:00:00"
#fi

#
# Script de submissão
#

if [ $(echo "$QSUB" | grep qsub) ]
then
  SCRIPTHEADER="
#PBS -j oe
#PBS -l walltime=${walltime}
#PBS -l mppwidth=${MPPWIDTH}
#PBS -l mppnppn=${MPPNPPN}
#PBS -l mppdepth=${MPPDEPTH}
#PBS -A CPTEC
#PBS -V
#PBS -S /bin/bash
${PBSDIRECTIVENAME}
${PBSDIRECTIVEARRAY}
#PBS -q ${QUEUE}
"
  SCRIPTRUNCMD="aprun -n ${MPPWIDTH} -N ${MPPNPPN} -d ${MPPDEPTH} -ss ${EXECFILEPATH}/ParModel_MPI < ${EXECFILEPATH}/MODELIN > ${EXECFILEPATH}/setout/Print.model.${LABELI}.MPI${MPPWIDTH}.log"
  SCRIPTRUNJOB="qsub -W block=true "
  SCRIPTMODULE="
export PBS_SERVER=${pbs_server1}
export HUGETLB_MORECORE=yes
export HUGETLB_ELFMAP=W
export HUGETLB_FORCE_ELFMAP=yes+
export MPICH_ENV_DISPLAY=1
export HUGETLB_DEFAULT_PAGE_SIZE=2m
export OMP_NUM_THREADS=6
"
  SCRIPTEXTRAS1="echo \${PBS_JOBID} > ${HOME_suite}/run/this.job.${LABELI}.${ANLTYPE}"
  SCRIPTEXTRAS2="sleep 10s # espera para terminar todos os processos de I/O"
else
  SCRIPTHEADER="
${PBSOUTFILE}
${PBSERRFILE}
#SBATCH --time=${WALLTIME}
#SBATCH --tasks-per-node=${MPPWIDTH}
#SBATCH --nodes=${MPPDEPTH}
${PBSDIRECTIVENAME}
${PBSDIRECTIVEARRAY}
#SBATCH --partition=${QUEUE}
"
  if [ $USE_SINGULARITY == true ]
  then          
    if [ ${ANLTYPE} != CTR -a ${ANLTYPE} != NMC -a ${ANLTYPE} != EIT -a ${ANLTYPE} != EIH ]
    then
      SCRIPTRUNCMD="module load singularity ; singularity exec -e --bind ${WORKBIND}:${WORKBIND} ${SIFIMAGE} mpirun -np ${MPPWIDTH} /usr/local/bin/ParModel_MPI < ${EXECFILEPATH}/\${MEM}${ANLTYPE:0:1}/MODELIN > ${EXECFILEPATH}/\${MEM}${ANLTYPE:0:1}/setout/Print.model.${LABELI}.MPI${MPPWIDTH}.log"
    else
      SCRIPTRUNCMD="module load singularity ; singularity exec -e --bind ${WORKBIND}:${WORKBIND} ${SIFIMAGE} mpirun -np ${MPPWIDTH} /usr/local/bin/ParModel_MPI < ${EXECFILEPATH}/MODELIN > ${EXECFILEPATH}/setout/Print.model.${LABELI}.MPI${MPPWIDTH}.log"
    fi        
  else  
    SCRIPTRUNCMD="mpirun -np ${MPPWIDTH} \${EXECFILEPATH}/ParModel_MPI < \${EXECFILEPATH}/MODELIN > \${EXECFILEPATH}/setout/Print.model.${LABELI}.MPI${MPPWIDTH}.log"
  fi  
  if [[ ! -z ${job_decanl_id} || ! -z ${job_deceof_id} ]]
  then        
    if [[ ${ANLTYPE} == "CTR" || ${ANLTYPE} == "RDP" ]]
    then
      SCRIPTRUNJOB="sbatch --dependency=afterok:${job_decanl_id}"
    elif [[ ${ANLTYPE} == "NMC" || ${ANLTYPE} == "NPT" || ${ANLTYPE} == "PPT" ]]
    then
      SCRIPTRUNJOB="sbatch --dependency=afterok:${job_deceof_id}"
    else
      SCRIPTRUNJOB="sbatch "
    fi
  else  
    SCRIPTRUNJOB="sbatch "
  fi        
  if [ $USE_INTEL == true ]
  then         
    SCRIPTMODULE="
# EGEON INTEL
module purge
module load ohpc
module swap gnu9 intel
module swap openmpi4 impi
module load hwloc
module load phdf5
module load netcdf
module load netcdf-fortran
module swap intel intel/2022.1.0

module list
"
  else
    if [ $USE_SINGULARITY == true ]
    then
      SCRIPTMODULE=""
    else      
      SCRIPTMODULE="
# EGEON GNU  
module purge
module load gnu9/9.4.0
module load ucx/1.11.2
module load openmpi4/4.1.1
module load netcdf/4.7.4
module load netcdf-fortran/4.5.3
module load phdf5/1.10.8
module load hwloc
module load libfabric/1.13.0

module list
"
    fi
  fi
  SCRIPTEXTRAS1=""
  SCRIPTEXTRAS2=""
fi

#monitor=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/setout/monitor.t
monitor=${EXECFILEPATH}/monitor.t
if [ -e ${monitor} ]; then rm ${monitor}; fi

cat <<EOF0 > ${SCRIPTFILEPATH}
#! /bin/bash -x
${SCRIPTHEADER}

${SCRIPTMODULE}

ulimit -s unlimited
ulimit -c unlimited

${PBSMEM}
${PBSEXECFILEPATH}

cd \${EXECFILEPATH}

${SCRIPTEXTRAS1}

date

${SCRIPTRUNCMD}

date

${SCRIPTEXTRAS2}

#touch ${monitor}
touch \${EXECFILEPATH}/monitor.t
EOF0

#
# Submete o script e aguarda o fim da execução
#

chmod +x ${SCRIPTFILEPATH}

job_model=$(${SCRIPTRUNJOB} ${SCRIPTFILEPATH})
export job_model_id=$(echo ${job_model} | awk -F " " '{print $4}')
echo "model ${job_model_id}"

if [ ${ANLTYPE} == CTR -o ${ANLTYPE} == NMC -o ${ANLTYPE} == EIT -o ${ANLTYPE} == EIH ]
then
  EXECFILEPATH=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}
  until [ -e ${EXECFILEPATH}/monitor.t ]; do sleep 1s; done
else
  for MEM in $(seq -f %02g 1 ${ANLPERT})
  do
    EXECFILEPATHMEM=${DK_suite}/model/exec_${PREFIC}${LABELI}.${ANLTYPE}/${MEM}${ANLTYPE:0:1}
    until [ -e ${EXECFILEPATHMEM}/monitor.t ]; do sleep 1s; done
  done
fi

if [ $(echo "$QSUB" | grep qsub) ]
then

  if [ ${ANLTYPE} != CTR -a ${ANLTYPE} != NMC -a ${ANLTYPE} != EIT -a ${ANLTYPE} != EIH ]
  then
  
    JOBID=$(cat ${HOME_suite}/run/this.job.${LABELI}.${ANLTYPE} | awk -F "[" '{print $1}')
  
    for mem in $(seq 1 ${ANLPERT})
    do
  
      jobidname="BAMENS${ANLTYPE}.o${JOBID}.${mem}"
      bamoutname="Out.model.${LABELI}.MPI${MPPWIDTH}.${mem}.out"
  
      until [ -e "${HOME_suite}/run/${jobidname}" ]; do sleep 1s; done
      mv -v ${HOME_suite}/run/${jobidname} ${EXECFILEPATH}/setout/${bamoutname}
    
    done
  
  else
  
    JOBID=$(cat ${HOME_suite}/run/this.job.${LABELI}.${ANLTYPE} | awk -F "." '{print $1}')
  
    jobidname="BAM${ANLTYPE}.o${JOBID}"
    bamoutname="Out.model.${LABELI}.MPI${MPPWIDTH}.out"
  
    until [ -e "${HOME_suite}/run/${jobidname}" ]; do sleep 1s; done 
    mv -v ${HOME_suite}/run/${jobidname} ${EXECFILEPATH}/setout/${bamoutname}
  
  fi
  
  rm ${HOME_suite}/run/this.job.${LABELI}.${ANLTYPE}

fi

#exit 0
