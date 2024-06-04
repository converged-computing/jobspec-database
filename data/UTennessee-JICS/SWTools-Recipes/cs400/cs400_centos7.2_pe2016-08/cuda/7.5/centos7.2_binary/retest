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

#-- Clean-out old tests:
rm -f ${SW_BLDDIR}/samples/bin/x86_64/linux/release/*

#-- Make some CUDA samples
cd ${SW_BLDDIR}/samples/7_CUDALibraries
for test in simpleCUBLAS simpleCUFFT cuSolverDn_LinearSolver batchCUBLAS; do
  echo "Making $test: "
  cd $test
  make clean
  make
  if [ $? -ne 0 ]; then
    echo "make $test failed"
    exit 1
  fi;
  cd ..
  echo ""
done

mkdir -p ${SW_WORKDIR}/cudaExec
cp ${SW_BLDDIR}/samples/bin/x86_64/linux/release/* \
   ${SW_WORKDIR}/cudaExec
   
cd ${SW_WORKDIR}

cat > ${PACKAGE}.pbs << EOF
#!/bin/bash
#PBS -N ${PACKAGE}
#PBS -j oe
#PBS -l nodes=1:nvidia:gpus=3

set -o verbose
cd \$PBS_O_WORKDIR

cd cudaExec

export PATH=${SW_BLDDIR}/bin:\$PATH
export LD_LIBRARY_PATH=${SW_BLDDIR}/lib64

TestPass=0
for iTest in \$( ls ); do
  echo "Running \$iTest:"
  ./\$iTest | tee -a ../${PACKAGE}.log | tee -a ${SW_BLDDIR}/.running
  if [ \$? -eq 0 ]; then
    echo "\$iTest:verified" >> ../status.all
  else
    echo "\$iTest:unverified" >> ../status.all
    TestPass=-1
  fi;
done

cd ..

if [ \$TestPass -eq 0 ]; then
  echo "verified" > ${SW_BLDDIR}/status
else
  echo "unverified" > ${SW_BLDDIR}/status
fi;
cat status.all >> ${SW_BLDDIR}/status

grep -A3 "Testing sgemm" ${PACKAGE}.log | grep GFLOPS \
  | awk '{print "YVALUE=",\$6}' | sed 's/GFLOPS=//g' | sed 's/ //g' \
  | split -l1 -a1 - sgemm.
cp sgemm.* ${SW_BLDDIR}
  
grep -A3 "Testing dgemm" ${PACKAGE}.log | grep GFLOPS \
  | awk '{print "YVALUE=",\$6}' | sed 's/GFLOPS=//g' | sed 's/ //g' \
  | split -l1 -a1 - dgemm. 
cp dgemm.* ${SW_BLDDIR}

JOBID=\`echo \$PBS_JOBID | cut -d "." -f1 \`
chmod 775 ${SW_BLDDIR}/status
rm ${SW_BLDDIR}/.running
cat ${PACKAGE}.o\${JOBID} >> ${SW_BLDDIR}/test.log
cat ${PACKAGE}.log >> ${SW_BLDDIR}/test.log
chmod 664 ${SW_BLDDIR}/test.log
EOF

#submit job and touch .running file - marker to infrastructure that
qsub ${PACKAGE}.pbs 2>&1 > ${SW_BLDDIR}/.running

# qsub returns 0 on successful job launch, so if failure return 1
if [ $? -ne 0 ]; then
  echo "Error submitting job"
  cat ${SW_BLDDIR}/.running
  rm -f ${SW_BLDDIR}/.running
  exit 1
else
  echo "Job submitted"
  cat ${SW_BLDDIR}/.running
  exit 2
fi

############################### if this far, return 0
exit 0
