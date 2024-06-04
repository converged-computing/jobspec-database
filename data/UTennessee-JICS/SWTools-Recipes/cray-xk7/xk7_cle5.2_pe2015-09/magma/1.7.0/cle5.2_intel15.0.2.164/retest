#!/bin/ksh

############################## standard interface to /sw tools
# Input:
#   Environment variables
#     SW_BLDDIR    current directory (PWD) minus /autofs/na1_ stuff
#     SW_ENVFILE   file to be sourced which has alternate prog environment
#                     only to be used in special circumstances
#     SW_WORKDIR   work dir that local script can use
# Output:
#   Return code of 0=success or 1=failure   or 2=job submitted
#
# Notes:
#   If this script is called from swtest, then swtest requires 
#   SW_WORKDIR to be set.  Then swtest adds a unique path to what 
#   user gave swtest (action+timestamp+build) and provides this
#   script with a uniquely valued SW_WORKDIR.  swtest will
#   automatically remove this unique workspace when retest is done.
##################################################################

# exit 3 is a signal to the sw infrastructure that this template has not 
# been updated; please delete it when ready
#exit 3

if [ -z $SW_BLDDIR ]; then
  echo "Error: SW_BLDDIR not set!"
  exit 1
else
  cd $SW_BLDDIR
fi

if [ -z $SW_ENVFILE ]; then
  ### Set Environment (do not remove this line only change what is in between)
  . ${MODULESHOME}/init/ksh
  . ${SW_BLDDIR}/remodule
  ### End Environment (do not remove this line only change what is in between)
else
  . $SW_ENVFILE
fi

############################## app specific section
#  

#clear out status file since re-testing
rm -f status 

cd ${SW_WORKDIR}
cp ${SW_SOURCES}/${PACKAGE}/testcases/testing_dgemm.cpp .

cat > ${PACKAGE}.pbs << EOF
#!/bin/bash
#PBS -N ${PACKAGE}
#PBS -j oe
#PBS -l nodes=1,walltime=01:00:00

set -o verbose
cd \$PBS_O_WORKDIR

module swap PrgEnv-pgi PrgEnv-intel
module load cudatoolkit magma/1.7.0

export MAGMA_TESTING=\$MAGMA_DIR/magma-1.7.0/testing

CC -O3 -fPIC -DADD_  -Wall -openmp -DHAVE_CUBLAS -DMIN_CUDA_ARCH=200 -DHAVE_CUBLAS -DMIN_CUDA_ARCH=200  -I\$MAGMA_INC  -I\$MAGMA_TESTING/../control -I\$MAGMA_TESTING/../sparse-iter/include -I\$MAGMA_TESTING -c testing_dgemm.cpp

CC -fPIC  testing_dgemm.o -o testing_dgemm \$MAGMA_TESTING/libtest.a  -lmagma -lcublas -lcudart
if [ \$? -ne 0 ] ; then
    echo "\$PACKAGE compile failed"
    exit 1
fi

aprun -n 1 ./testing_dgemm -c | tee -a ${SW_BLDDIR}/.running > magma.dgemm.out

testspassed=\`grep "ok" magma.dgemm.out | wc -l \`
if [[ \$testspassed -ne 10 ]]; then
  echo unverified > $SW_BLDDIR/status
else
  echo verified > $SW_BLDDIR/status
fi
JOBID=\`echo \$PBS_JOBID | cut -d "." -f1 \`
chmod 775 ${SW_BLDDIR}/status
rm ${SW_BLDDIR}/.running
cat \${JOBID}.OU >> ${SW_BLDDIR}/test.log
cat magma.dgemm.out >> ${SW_BLDDIR}/test.log
chmod 664 ${SW_BLDDIR}/test.log
EOF
#submit job and touch .running file - marker to infrastructure that
qsub $PACKAGE.pbs > ${SW_BLDDIR}/.running

# qsub returns 0 on successful job launch, so if failure return 1
if [ $? -ne 0 ]; then
  echo "Error submitting job"
  rm -f .running
  exit 1
else
  echo "Job submitted"
  cat ${SW_BLDDIR}/.running
  exit 2
fi



cd ../

############################### if this far, return 0
exit 0
