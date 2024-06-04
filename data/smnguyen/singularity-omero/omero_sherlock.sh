#!/bin/bash -l

#SBATCH --partition=mcovert
#SBATCH --time=23:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#
# Note: the time for the job is shorter than 24 hours to prevent multiple instances
# of OMERO running concurrently.
#
# 01 00 * * * /share/PI/mcovert/omero/share_job.sh /share/PI/mcovert/etc/share_job/omero /share/PI/mcovert/omero/sherlock.sh 24064 28080

function get_port {
  local DESIRED_PORT=$1
  while [ $(netstat -lnt | grep ":$DESIRED_PORT " | wc -l) -ne 0 ]; do
    (( DESIRED_PORT++ ))
  done
  echo $DESIRED_PORT
}

PGPASSWORD=covert_omero
cd /share/PI/mcovert/omero
module load singularity/2.2

set -x

PG_PORT=$(get_port 25432)
REGISTRY_PORT=$(get_port 24061)
TCP_PORT=$(get_port 24063)
SSL_PORT=$(get_port 24064)
WEB_PORT=$(get_port 24080)
HTTP_PORT=$(get_port 28080)

exec singularity run -w -B "./data:/omero/data" -B "./var:/omero/var" -B "./postgres:/omero/postgres" -B "./run:/omero/run" -B "./user_scripts:/omero/user_scripts" omero.img --pg-port $PG_PORT --pg-password $PGPASSWORD --server-registry-port $REGISTRY_PORT --server-tcp-port $TCP_PORT --server-ssl-port $SSL_PORT --web $WEB_PORT --http $HTTP_PORT
