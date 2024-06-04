#! /bin/bash
################################################################################
# build.sh: Compile iorun using the mpi/OpenMPI module
# Time-stamp: <Thu 2014-01-28 18:36 hcartiaux>
#
# Copyright (c) 2014 Hyacinthe Cartiaux <Hyacinthe.Cartiaux@uni.lu>
################################################################################

##########################
#                        #
#   The OAR  directives  #
#                        #
##########################
#
#          Set number of resources
#

#OAR -l core=1,walltime=0:10:0

#          Set the name of the job (up to 15 characters,
#          no blank spaces, start with alphanumeric character)

#OAR -n IORUN_COMPIL

#          By default, the standard output and error streams are sent
#          to files in the current working directory with names:
#              OAR.%jobid%.stdout  <-  output stream
#              OAR.%jobid%.stderr  <-  error stream
#          where %job_id% is the job number assigned when the job is submitted.
#          Use the directives below to change the files to which the
#          standard output and error streams are sent, typically to a common file

#OAR -O IORUN_COMPIL-%jobid%.log
#OAR -E IORUN_COMPIL-%jobid%.log

#####################################

if [ -f  /etc/profile ]; then
    .  /etc/profile
fi

module load mpi/OpenMPI

cd ./src/ior
./bootstrap
export LIBS="${LIBS} -L/usr/lpp/mmfs/lib/"
./configure
make

cd ../..

ln -sf ../src/ior/src/ior runs/ior

