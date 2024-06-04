#!/bin/bash
#PBS -o zuphux.cerit-sc.cz:logs/$$.stdout
#PBS -e zuphux.cerit-sc.cz:logs/$$.stderr
#PBS -l select=1:ncpus=$CPUL:mem=$MEML:ngpus=$GPUR:scratch_ssd=20gb:vnode=^samson

sshkey="$ssh_key"

home=`pwd`

echo "$sshkey" > ${home}/id_rsa.$$
chmod 0600 ${home}/id_rsa.$$

image=$CONTAINER

mounts="$MNT"

cache="${home}/cache"
sif=`echo $image | sed -e 's/\//-/g'`

mkdir $cache 2> /dev/null

if [ -z $SCRATCHDIR ]; then
  echo "SCRATCHDIR env must be defined"
  exit 1
fi

cd $SCRATCHDIR || exit 1

export TMPDIR=$SCRATCHDIR


while ! mkdir "$cache/$sif.lck" 2> /dev/null; do
        sleep 1;
done

if ! [ -f "$cache/$sif" ]; then
        singularity pull "$cache/$sif" "docker://$image"
fi

rmdir "$cache/$sif.lck"

j=1;
for i in $mounts; do
  mkdir "$j"
  k=1;
  while true; do
     sshfs -o IdentityFile=${home}/id_rsa.$$,UserKnownHostsFile=/dev/null,StrictHostKeyChecking=no -p ${ssh_port} ${USER_NAME}@${ssh_host}:"$i" "$j" && break;
     echo "Waiting for ssh endpoint to be ready"
     sleep $k;
     k=$[k*2];
     if [ $k -gt 2048 ]; then
	     echo "Timeout waiting for ssh endpoint";
             exit 2;	     
     fi
  done
  binds=(${binds[*]} '--bind' "$j:$i")
  j=$[j+1]
done

$ENVS

singularity run --bind $SCRATCHDIR:$SCRATCHDIR ${binds[*]} -i "$cache/$sif" $CMD

ret=$?

j=$[j-1]

for i in `seq 1 $j`; do
        umount "$i" && rmdir "$i";
done

rm -f ${home}/id_rsa.$$

exit $ret
