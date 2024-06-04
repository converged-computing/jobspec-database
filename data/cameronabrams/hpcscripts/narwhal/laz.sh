#!/bin/bash
fileset=()
for f in *-r1.log; do
#   echo $f
   sz=`wc -c $f|cut -f1 -d' '`
#   echo $sz
   if [[ $sz < 4400000 ]]; then
      p=`echo $f | cut -f1 -d'-'`
      /home/cfa/Git/psfgen/scripts/prep_namd_restart.sh -l ${p}.log -i ${p}.namd -o ${p}-r1.namd --addsteps 10000000
      for f in `cat ${p}-r1.namd.files`; do
         if [[ ! " ${fileset[*]} " =~ " ${f} " ]]; then
            fileset+=($f)
         fi
      done
      cat > us-${p}-r1.js << EOF
#!/bin/bash
#PBS -A ARLAP35100034
#PBS -q standard
#PBS -l select=16:ncpus=44:mpiprocs=44
#PBS -l walltime=18:00:00
#PBS -N ${p}-r1
#PBS -j oe
#PBS -V
#PBS -S /bin/bash
#PBS -m be
#PBS -M cfa22@drexel.edu

cd \$PBS_O_WORKDIR

PLAT=Linux-x86_64-g++
CHARMRUN=/p/home/cfabrams/apps/namd/NAMD_2.14_Source/\${PLAT}/charmrun
NAMD2=/p/home/cfabrams/apps/namd/NAMD_2.14_Source/\${PLAT}/namd2

\$CHARMRUN +p704 \$NAMD2 ${p}-r1.namd > ${p}-r1.log
EOF
      fileset+=(us-${p}-r1.js)
   fi
done

tar cf "reruns.tar" ${fileset[@]}
