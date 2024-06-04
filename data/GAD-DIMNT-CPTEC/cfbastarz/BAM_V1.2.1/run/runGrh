#! /bin/bash 

#help#
#-------------------------------------------------------------------------------------------------#
#                         Brazilian global Atmospheric Model - BAM_V1.2.1                         #
#-------------------------------------------------------------------------------------------------#
# Descrição:                                                                                      #
#     Script executar o grid-history do modelo BAM                                                #
#                                                                                                 #
# Uso:                                                                                            #
#     ./runGrh cpu_mpi cpu_node task_omp TRC LV LABELI LABELF name PS PREFIXC hold                #
#                                                                                                 #
# Exemplo:                                                                                        #
#     ./runGrh 126 28 2016021100 2013010100 BAMGRH psurf NMC hold                                 #
#                                                                                                 #
# Opções:                                                                                         #
#     * cpu_mpi.: número de processos MPI (eg., 48, 96, 192, 384)                                 #
#     * cpu_node: número de processos MPI por nós de memória compartilhada (eg., 4)               #
#     * task_omp: número de tarefas OpenMP (eg., 6)                                               #
#     * name....: nome do job (eg., BAMPOST)                                                      #
#     * TRC.....: resolução espectral das previsões (eg., 62, 126, 213, 299, 666)                 #
#     * LV......: resolução vertical das previsões (eg., 28, 42, 64, 96)                          #
#     * LABELI..: data da análise (eg., 2013010100)                                               #
#     * LABELF..: data final das previsões (eg., 2013010300)                                      #
#     * PS......: tipo de pressão (eg., psurf)                                                    #
#     * PREFIXC.: prefixo das previsões (eg., NMC, CPT, AVN)                                      #
#     * hold....: segura o script até que a execução do job termine (opcional, indicar ou não)    #
#                                                                                                 #
# DMD/CPTEC/INPE, 2019                                                                            #
#-------------------------------------------------------------------------------------------------#
#help#

#
# Função para imprimir o cabeçalho do script
#
function usageprincipal()
{
  cat < ${0} | sed '1,/^#help#/d;/^#help#/,$d'
}

#
# Verifica o número de argumentos passados junto com o script:
# pelo menos 7 argumentos devem ser passados;
# caso contrário, será chamada a função para imprimir o cabeçalho
# e o script é encerrado
#
if [ "$#" -lt 7 ]
then
  usageprincipal
  exit 0
fi

if [ -z "${1}" ]
then
  echo "TRC is not set"
else
  TRC=$(echo ${1} | awk '{print $1/1}')
fi

if [ -z "${2}" ]
then
  echo "LV is not set" 
  exit 2
else
  LV=$(echo ${2} | awk '{print $1/1}')
fi

if [ -z "${3}" ]
then
  echo "LABELI is not set" 
  exit 3
else
  export LABELI=${3}  
fi

if [ -z "${4}" ]
then
  echo "LABELF is not set" 
  exit 3
else
  export LABELF=${4}  
fi

if [ -z "${5}" ]
then
  echo "NAME is not set" 
  exit 3
else
  export NAME=${5}  
fi

if [ -z "${6}" ]
then
  echo "PS is not set" 
  exit 3
else
  export PS=${6}  
fi

if [ -z "${7}" ]
then
  echo "PREFIXC is not set" 
  exit 3
else
  export PREFIXC=${7}  
fi

#
# Verifica se a opção "hold" foi
# passada pela linha de comando
#
export hold=""
for arg in $*
do
  if test ${arg} = "hold"; then export hold="-h"; fi
done

if [ ${TRC} = 21 ];   then export timestep=3600; fi 
if [ ${TRC} = 31 ];   then export timestep=1800; fi 
if [ ${TRC} = 42 ];   then export timestep=1800; fi 
if [ ${TRC} = 62 ];   then export timestep=1200; fi
if [ ${TRC} = 106 ];  then export timestep=900;  fi
if [ ${TRC} = 126 ];  then export timestep=600;  fi
if [ ${TRC} = 133 ];  then export timestep=600;  fi
if [ ${TRC} = 159 ];  then export timestep=600;  fi
if [ ${TRC} = 170 ];  then export timestep=450;  fi
if [ ${TRC} = 213 ];  then export timestep=450;  fi
if [ ${TRC} = 213 ];  then export timestep=360;  fi
if [ ${TRC} = 254 ];  then export timestep=300;  fi
if [ ${TRC} = 299 ];  then export timestep=300;  fi
if [ ${TRC} = 319 ];  then export timestep=225;  fi
if [ ${TRC} = 341 ];  then export timestep=200;  fi
if [ ${TRC} = 382 ];  then export timestep=180;  fi
if [ ${TRC} = 511 ];  then export timestep=150;  fi
if [ ${TRC} = 533 ];  then export timestep=150;  fi
if [ ${TRC} = 666 ];  then export timestep=240;  fi
if [ ${TRC} = 863 ];  then export timestep=150;  fi
if [ ${TRC} = 1279 ]; then export timestep=20;   fi

#
# Ajuste das variáveis de ambiente
#
CASE=$(echo ${TRC} ${LV} | awk '{ printf("TQ%4.4dL%3.3d\n",$1,$2)  }')
PATHA=$(pwd)

export FILEENV=$(find -L ${PATHA} -name EnvironmentalVariablesMCGA -print)
export PATHENV=$(dirname ${FILEENV})
export PATHBASE=$(cd ${PATHENV}; cd ../; pwd)

. ${FILEENV} ${CASE} ${PREFIXC}

cd ${HOME_suite}/run

#
# Ajuste dos caminhos e diretórios 
#
export DIRRESOL=$(echo ${TRC} ${LV} | awk '{ printf("TQ%4.4dL%3.3d\n",$1,$2)  }')
export EXECFILEPATH=${DK_suite}/grh/exec
export EXECFILEPATH2=${DK_suite}/grh/exec_${PREFIC}${LABELI}
export SCRIPTFILEPATH=${EXECFILEPATH2}/modg.${PREFIXC}.${DIRRESOL}.${MAQUI}
export NAMELISTFILEPATH=${HOME_suite}/run; mkdir -p ${HOME_suite}/run/setout
export OUTPUTFILEPATH=${HOME_suite}/run/setout/grhg.${PREFIXC}.${DIRRESOL}.${MAQUI}.${RUNTM}.out
export GRHDATAOUT=${GRHDATAOUT}/${DIRRESOL}/${LABELI}

mkdir -p ${EXECFILEPATH2}/setout

mkdir -p ${GRHDATAOUT}/AC/; mkdir -p ${GRHDATAOUT}/AL/; mkdir -p ${GRHDATAOUT}/AM/;
mkdir -p ${GRHDATAOUT}/AP/; mkdir -p ${GRHDATAOUT}/BA/; mkdir -p ${GRHDATAOUT}/CE/;
mkdir -p ${GRHDATAOUT}/DF/; mkdir -p ${GRHDATAOUT}/ES/; mkdir -p ${GRHDATAOUT}/GO/;
mkdir -p ${GRHDATAOUT}/MA/; mkdir -p ${GRHDATAOUT}/MG/; mkdir -p ${GRHDATAOUT}/MS/;
mkdir -p ${GRHDATAOUT}/MT/; mkdir -p ${GRHDATAOUT}/PA/; mkdir -p ${GRHDATAOUT}/PB/;
mkdir -p ${GRHDATAOUT}/PE/; mkdir -p ${GRHDATAOUT}/PI/; mkdir -p ${GRHDATAOUT}/PR/;
mkdir -p ${GRHDATAOUT}/RJ/; mkdir -p ${GRHDATAOUT}/RN/; mkdir -p ${GRHDATAOUT}/RO/;
mkdir -p ${GRHDATAOUT}/RR/; mkdir -p ${GRHDATAOUT}/RS/; mkdir -p ${GRHDATAOUT}/SC/;
mkdir -p ${GRHDATAOUT}/SE/; mkdir -p ${GRHDATAOUT}/SP/; mkdir -p ${GRHDATAOUT}/TO/;
mkdir -p ${GRHDATAOUT}/WW/; mkdir -p ${GRHDATAOUT}/ZZ/; 

#
# Atribuição do número de processadores
#
cpu_mpi=${1};  if [[ -z "${1}"  ]]; then cpu_mpi=1 ; fi
cpu_node=${2}; if [[ -z "${2}"  ]]; then cpu_node=1; fi

export cpu_mpi cpu_node
export cpu_node
export RES=${3}

num=$(($cpu_mpi+$cpu_node-1))
fra=$(($num/$cpu_node))
cpu_tot=$(($fra*$cpu_node))

echo "fila=mpi-npn${cpu_node} total cpus=${cpu_tot}"

export PBS_SERVER=aux20-eth4

host=$(hostname)

#
# Grads
#
export gradsbin="grads"
export trunc=${TRC}
export lev=${LV}
export aspa="'"

if [ -z "${PREFXO}" ]
then
  export Preffix="NMC" 
else
  export Preffix="${PREFXO}" 
fi
 
export DirInPut="${DK_suite2}/model/dataout/${DIRRESOL}/${LABELI}/"; mkdir -p  ${DirInPut}  
export DirOutPut="${DK_suite2}/pos/dataout/${DIRRESOL}/${LABELI}/"; mkdir -p  ${DirOutPut}  
export DirMain="${DK_suite}"  

export TMean=3600

labeli=${LABELI}
labelf=${LABELF}
name=${NAME}
ps=${PS}

MAQUI=$(uname -s)

if [ -z "${ps}" ]
then
  ps=psuperf
fi

ext=$(echo ${TRC} ${LV} |awk '{ printf("TQ%4.4dL%3.3d\n",$1,$2)  }')
DATE=$(echo $LABELI | cut -c1-8)
HH=$(echo $LABELI | cut -c9-10)
DATEF=$(echo $LABELF | cut -c1-8)
HHF=$(echo $LABELF | cut -c9-10)
labelr=$(date -d "${DATE} ${HH}:00 12 hour ago" +"%Y%m%d%H")
julday1=$(date -d "${DATE} ${HH}:00" +"%j")
julday2=$(date -d "${DATEF} ${HHF}:00" +"%j")

ndays=$(echo ${julday2} ${julday1} |awk '{{nday=$1-$2}if(nday < 0){nday = $1 + (365-$2)} if(nday >7){nday=7} {print nday}}')

#
# Construção do namelist do grid history
#
cat ${NAMELISTFILEPATH}/PostGridHistory.nml | awk '{  
      if (substr($1,1,4) == "Mend")
       {
     	"echo $trunc" | getline trunc
         printf("Mend=%d,               ! Model Spectral Horizontal Resolution\n",trunc)
       }
      else if (substr($1,1,4) == "Kmax")
       {
     	"echo $lev" | getline lev      
         printf("Kmax=%d,                ! Number of Vertical Model Layers\n",lev)
       }
      else if (substr($1,1,4) == "DelT")
       {
     	"echo $timestep" | getline timestep      
         printf("DelT=%d,         ! Model Time Step in Seconds\n",timestep)
       }
      else if (substr($1,1,5) == "TMean")
       {
     	"echo $TMean" | getline TMean      
         printf("TMean=%d,        ! Time Interval in Seconds To Average Output (1 Hour)\n",TMean)
       }
      else if (substr($1,1,6) == "LabelI")
       {
     	"echo $LABELI" | getline LABELI       
    	"echo $aspa" | getline aspa    
         printf("LabelI=%s%s%s,     ! Initial Condition Date\n",aspa,LABELI,aspa)
       }
      else if (substr($1,1,6) == "LabelF")
       { 
     	"echo $LABELF" | getline LABELF
    	"echo $aspa" | getline aspa    
         printf("LabelF=%s%s%s,     ! Final Forecast Date\n",aspa,LABELF,aspa)
       }
      else if (substr($1,1,7) == "Preffix")
       { 
     	"echo $Preffix" | getline prefx
    	"echo $aspa" | getline aspa    
         printf("Preffix=%s%s%s,          ! Preffix of File Names\n",aspa,prefx,aspa)
       }
      else if (substr($1,1,8) == "DirInPut")
       { 
     	"echo $DirInPut" | getline DirInPut
    	"echo $aspa" | getline aspa    
         printf("DirInPut=%s%s%s, ! Main Data Directory\n",aspa,DirInPut,aspa)
       }
      else if (substr($1,1,9) == "DirOutPut")
       { 
     	"echo $DirOutPut" | getline DirOutPut
    	"echo $aspa" | getline aspa    
         printf("DirOutPut=%s%s%s, ! Main Data Directory\n",aspa,DirOutPut,aspa)
       }
      else if (substr($1,1,5) == "DirMain")
       { 
     	"echo $DirMain" | getline DirMain
    	"echo $aspa" | getline aspa    
         printf("DirMain=%s%s%s,    ! Main Data Directory\n",aspa,DirMain,aspa)
       }
       else
       {
     	 print $0
       }
     }' > ${EXECFILEPATH2}/PostGridHistory.nml

RUNTM=$(date +'%Y%m%d%T')

# 
# Script de submissão
#
dateoutjob=$(date +"%Y%m%d%H%S")

cat << EOT > ${SCRIPTFILEPATH}
#!/bin/bash
#PBS -o ${EXECFILEPATH2}/setout/Out.grh.${PREFIXC}.${labeli}.${tmstp}.%s.MPI${cpu_mpi}.out
#PBS -j oe
#PBS -l walltime=00:10:00
#PBS -lselect=1:ncpus=1
#PBS -V
#PBS -S /bin/bash
#PBS -N ${RES}
#PBS -q ${AUX_QUEUE}

export PBS_SERVER=aux20-eth4

if [[ ${MAQUI} == "Linux" || ${MAQUI} == "linux" ]]
then
  export F_UFMTENDIAN=10,20,30,40,50,60,70,80
  export GFORTRAN_CONVERT_UNIT=big_endian:10,20,30,40,50,60,70,80
fi

export KMP_STACKSIZE=128m

ulimit -s unlimited

cp -f ${EXECFILEPATH}/PostGridHistory ${EXECFILEPATH2}/

cd ${EXECFILEPATH2}

mkdir -p setout

date

time ${EXECFILEPATH2}/PostGridHistory < ${EXECFILEPATH2}/PostGridHistory.nml

date

echo "" > ${EXECFILEPATH2}/monitor.grh
EOT

#
# Submissão do script
#
chmod +x ${SCRIPTFILEPATH}

cd ${EXECFILEPATH2}

export PBS_SERVER=aux20-eth4

#qsub -W block=true ${SCRIPTFILEPATH}
${QSUB} ${hold} ${SCRIPTFILEPATH}

until [ -e ${EXECFILEPATH2}/monitor.grh ]; do sleep 1s; done

rm -fr ${EXECFILEPATH2}/monitor.grh

echo "LABELI = ${labeli} LABELF = ${labelf} LABELR = ${labelr}"
echo "PARAMETROS GRADS ==> ${labeli} ${labelf} ${name} ${ext} ${ps} ${labelr}"

cd  ${GRHDATAOUT}

rm -f ${GRHDATAOUT}/umrs_min??????????.txt

#
# Christopher - 24/01/2005
# OBS: O GrADS script abaixo e quem inicializa/cria o arquivo deltag.${labeli}.out
#
echo "${labeli} ${labelf} ${name} ${ext} ${ps} ${labelr}"

cat << EOF1 > ${HOME_suite}/grh/scripts/meteogr.gs
'reinit'

pull argumentos

_labeli=subwrd(argumentos,1);_labelf=subwrd(argumentos,2);_nomea=subwrd(argumentos,3);_trlv=subwrd(argumentos,4)
_ps=subwrd(argumentos,5);_labelr=subwrd(argumentos,6)
_time1=subwrd(argumentos,7);_time2=subwrd(argumentos,8);_prefix=subwrd(argumentos,9)

* nome do arquivo com prefixos dos pontos
_nomeb='${DK_suite2}/pos/dataout/${DIRRESOL}/${labeli}/Preffix'%_labeli%_labelf%'.'%_trlv

* nomes dos arquivos com identificacao e local dos pontos
_nomec='${DK_suite2}/pos/dataout/${DIRRESOL}/${labeli}/Identif'%_labeli%_labelf%'.'%_trlv
_nomed='${DK_suite2}/pos/dataout/${DIRRESOL}/${labeli}/Localiz'%_labeli%_labelf%'.'%_trlv
nomectl ='${DK_suite2}/pos/dataout/${DIRRESOL}/${labeli}/'%_nomea%_labeli%_labelf%'M.grh.'%_trlv%'.ctl'
*say nomectl

_lonlat2ur="05038W2104S 04823W2137S 04823W2030S 04856W2211S 04715W2245S 04715W2030S 05004W2211S 04749W2245S 05111W2211S 04749W2104S 04930W2104S 04715W2318S"
_nlonlat2ur=12

_ndias=${ndays}
_ntimes=_ndias*24
say "abrindo o arquivo "nomectl
'open 'nomectl
'q file'
say result

say _time1' '_time2
'set time '_time1' '_time2

rec=read(_nomeb);nloc=sublin(rec,2)
rec=read(_nomec);nloc1=sublin(rec,2)
rec=read(_nomed);nloc2=sublin(rec,2)
_loc=0

while (_loc < nloc)
   _loc=_loc+1
   say 'nloc ' nloc ' - ' _loc
   'clear';'set x '_loc
   'set time '_time1' '_time2
   routine=dimet()
   if (_faz=1)
      lixo=write('umrs_min'_labeli'.txt',_linha)
      lixo=write('umrs_min'_labeli'.txt','')
   endif  
endwhile
lixo=close('umrs_min'_labeli'.txt')

***********************************************
********* Grid History Maps Finalized *********
***********************************************
function dimet()
'set display color white';'clear'
'set vpage 0 8.5 10.2 11';'set grads off'
routine=title()
'set vpage 0 8.5 8.75 10.70';'set parea 0.7 8.0 0.3 1.6';'set grads off'
routine=prec()
'set vpage 0 8.5 7.00 8.95';'set parea 0.7 8.0 0.3 1.6';'set grads off'
routine=temp()
'set vpage 0 8.5 5.25 7.20';'set parea 0.7 8.0 0.3 1.6';'set grads off'
routine=umrl()
if (_faz=1); routine=umrl_min(); endif 
'set vpage 0 8.5 3.50 5.45';'set parea 0.7 8.0 0.3 1.6';'set grads off'
routine=wvel()
'set vpage 0 8.5 1.75 3.70';'set parea 0.7 8.0 0.3 1.6';'set grads off'
if (_ps=reduzida)
routine=psnm()
else
routine=pslc()
endif
'set vpage 0 8.5 0.0 1.95';'set parea 0.7 8.0 0.3 1.6';'set grads off'
routine=cbnv()
label=_labeli%_labelf
rec=read(_nomeb);lab=sublin(rec,2)
taga=png;tagb=png;tagc=png
say 'printim ${GRHDATAOUT}/'lab'.png png'
'printim ${GRHDATAOUT}/'lab'.png png' 
'!rm -f meteogram'
if (_loc=1)
'!rm -f puttag.'_labeli'.out'
'!rm -f deltag.'_labeli'.out'
endif
*AAF '!echo put '_state'/'lab''label'.'taga' >> puttag.'_labeli'.out'
return
************************************************
function title()
rec=read(_nomec);local=sublin(rec,2)
_state=estado(local)
rec=read(_nomed)
lonlat=sublin(rec,2);loi=substr(lonlat,1,3);lof=substr(lonlat,4,2)
lo=substr(lonlat,6,1);lai=substr(lonlat,7,2);laf=substr(lonlat,9,2);la=substr(lonlat,11,1)
lalo=loi%':'%lof%lo%'-'%lai%':'%laf%la
say ' '
say '   Plotando localizacao numero = '_loc'  Local: 'local
'set string 8 l 6';'set strsiz .13 .14'
'draw string 0.4 0.7 CPTEC:'
'draw string 1.4 0.7 'lalo
'draw string 3.4 0.7 'local
_faz=0

n=1
while (n<=_nlonlat2ur)
   latlonur=subwrd(_lonlat2ur,n)
   if (lonlat=latlonur)
      lixo=write('umrs_min'_labeli'.txt',lonlat' - 'local)
      _faz=1
      _linha=""
      n=9999
   endif
   n=n+1
endwhile

'query files'
lbin=sublin(result,3);bin=subwrd(lbin,2);utc=substr(bin,67,2)
'set t 5';'q dims'
tm = sublin(result,5);tim = subwrd(tm,6);tmm = substr(tim,7,9)
if (_ps!=reduzida)
'd topo'
_tpg=subwrd(result,4)
endif
'set strsiz .125 .13'
'draw string 0.4 0.5 'tmm' 'utc'Z (GMT)                           Vertical Grid Line: 'utc'Z'
'set time '_time1' '_time2
return

************************************************
function umrl()
'set gxout line';'set grads off';'set axlim 0 100';'set cmark 0';'set ylint 20';'set ccolor 4'
'd umrs'
'set string 6 l 5';'set strsiz .12 .13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
'draw string 1 1.75 Relative Humidity (%)'
return
************************************************
function umrl_min()
t=1;urmin=200; datamin=xx
while (t<=_ntimes)
'set t 't''
'q time'
data=subwrd(result,3)
'd umrs'
umid=subwrd(result,4)
if (umid<urmin)
urmin=umid
datamin=data
endif  
  
fimdia=math_fmod(t,24)
if (fimdia=0)
urmin=math_format('%5.1f',urmin)
_linha=_linha' 'datamin' 'urmin
urmin=200; datamin=xx
endif
t=t+1
endwhile
'set time '_time1' '_time2
return
************************************************
function cbnv()
'set gxout bar';'set bargap 0';'set barbase 0';'set vrange 0 100';'set ylint 20';'set grads off';'set ccolor 15'
'd cbnv'
'set string 6 l 5';'set strsiz 0.12 0.13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
'draw recf 0 0 8.5 0.1'
'draw string 1 1.75 Cloud Cover (%)'
return
************************************************
function snof()
'set gxout bar';'set bargap 0';'set barbase 0';'set vrange 0 10';'set ylint 2';'set grads off';'set ccolor 4'
'd neve'
'set string 6 l 5';'set strsiz 0.12 0.13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
'draw string 1 1.75 Snow Fall (mm/h)'
return
************************************************
function prec()
'set gxout bar';'set bargap 0';'set barbase 0';'set vrange 0 5';'set ylint 1';'set grads off';'set ccolor 4'
'd prec'
'set gxout stat';'d neve'
lnv=sublin(result,8)
nv=subwrd(lnv,5)
'set gxout bar';'set string 6 l 5';'set strsiz 0.12 0.13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
if (nv > 0.0001)
'set ccolor 3'
'd neve'
'draw string 1 1.75 Precipitation (blue) and Snow Fall (green) (mm/h)'
else
'draw string 1 1.75 Precipitation (mm/h)'
endif
'set string 8 l 6';'set strsiz .13 .14'
'draw string 7.1 1.75 '_trlv
return
************************************************
function psnm()
rotina=maxmin(psnm)
'set gxout line';'set vrange '_vmin' '_vmax'';'set cmark 0';'set ylint '_itvl'';'set ccolor 4';'set grads off'
'd psnm'
'set string 6 l 5';'set strsiz 0.12 0.13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
'draw string 1 1.75 Mean Sea Level Pressure (hPa)'
return
************************************************
function pslc()
rotina=maxmin(pslc)
'set gxout line';'set vrange '_vmin' '_vmax'';'set cmark 0';'set ylint '_itvl'';'set ccolor 4';'set grads off'
'd pslc'
'set string 6 l 5';'set strsiz 0.12 0.13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
'draw string 1 1.75 Surface Pressure (hPa)        Model Altitude: '_tpg' m'
return
************************************************
function wvel()
'set lev 1000 ';'set gxout vector';'set ylab off';'set grads off';'set arrowhead 0.075';'set z 0.5 1.5'
'd skip(uves,1,12);vves'
'set gxout line';'set grads off';'set z 1';'set ylab on';'set cmark 0';'set ylint 2';'set ccolor 4'
'd mag(uves,vves)'
'set string 6 l 5';'set strsiz .12 .13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
'draw string 1 1.75 Surface Wind (m/s)'
return
************************************************
function temp()
rotina=maxmin(tems)
'set gxout line';'set vrange '_vmin' '_vmax'';'set grads off';'set ylint '_itvl'';'set ccolor 4';'set cmark 0'
'd tems'
'set string 6 l 5';'set strsiz .12 .13';'set line 0'
'draw recf 0.75 1.65 8.5 1.82'
'draw string 1 1.75 Surface Temperature (\`aO\`nC)'
return
************************************************
function maxmin(var)
'set t 1'

'd max('var',t=1,t='_ntimes',1)'

linha=sublin(result,2);imax=subwrd(linha,4)
'd min('var',t=1,t='_ntimes',1)'

linha=sublin(result,2);imin=subwrd(linha,4)
say linha
say imin

if(imin>0);imin=imin-1;else;imin=imin+1;endif
if(imax>0);imax=imax+1;else;imax=imax-1;endif
_vmax=math_nint(imax);_vmin=math_nint(imin)
_itvl=math_nint((imax-imin)/5)
'set time '_time1' '_time2
return
************************************************
function estado(local)
frase='AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO'
ne=1
while(ne<=27)
est.ne=subwrd(frase,ne)
ne=ne+1
endwhile

i=1
c=substr(local,i,1)
while (c != '(')
i=i+1
c=substr(local,i,1)
if (i > 40)
break
endif
endwhile
j=1
c=substr(local,j,1)
while (c != ')')
j=j+1
c=substr(local,j,1)
if (j > 40)
break
endif
endwhile
if (i > 40 | j > 40)
state='ZZ'
else
i=i+1
j=j-i
state=substr(local,i,j)
k=0
l=0
while (k < 27)
k=k+1
if (state = est.k)
l=1
endif
endwhile
endif
if (l = 0)
state='WW'
endif
return state
***********************************************
EOF1

DATE=$(echo ${LABELI} | cut -c 1-8)
HH=$(echo ${LABELI} | cut -c 9-10)
DATEF=$(echo ${LABELF} | cut -c 1-8)
HHF=$(echo ${LABELF} | cut -c 9-10)

time1=$(date -d "$DATE $HH:00" +"%HZ%d%b%Y")
time2=$(date -d "$DATEF $HHF:00" +"%HZ%d%b%Y")

echo "${labeli} ${labelf} ${name} ${ext} ${ps} ${labelr}"
echo "${time1} ${time2}"

if [ $GSSTEP = 9999 ] # PARA LIGAR TROCAR POR 1 AAF
then
/opt/grads/2.0.a9/bin/grads -bp << EOT
run ${HOME_suite}/grh/scripts/meteogr.gs
${labeli} ${labelf} ${name} ${ext} ${ps} ${labelr} ${time1} ${time2} ${PREFIXC}
quit
EOT
  echo "run ${HOME_suite}/grh/scripts/meteogr.gs"
  echo "${labeli} ${labelf} ${name} ${ext} ${ps} ${labelr} ${time1} ${time2} ${PREFIXC}"
fi

exit 0
