#!/bin/bash

#OAR -l /nodes=1/core=8,walltime=11:59:00
### OAR -p virt='YES' {mem_core > 1024}
### OAR -p cluster='manda'
#OAR -O /temp_dd/igrida-fs1/psaffray/oar_output/job.%jobid%.output
#OAR -E /temp_dd/igrida-fs1/psaffray/oar_output/job.%jobid%.error

. /etc/profile.d/modules.sh
set -x
module load veertuosa/0.0.1

VM_NAME=TuxML_${OAR_JOBID}

veertuosa_launch --name ${VM_NAME} --image /temp_dd/igrida-fs1/psaffray/images/ubuntu-18.04-x86_64_clean.qcow2

if [ $1 = "tuxml" ]
then
	VM_CMD="~/launch_test.sh"
else
	VM_CMD="~/launch_make.sh"
fi
### VM_CMD="~/kernel_generator.py --local --dev --config test.config --linux4_version 15 5000"

ssh-vm $VM_NAME "$VM_CMD $OAR_JOB_ID;/sbin/poweroff"
