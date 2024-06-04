#!/bin/bash

# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Specify partition
#SBATCH -p dgxa100-mig
 
# Request a single GPU
#SBATCH -G 1
 
# Request 4 CPU tasks
#SBATCH -n 1

GID=$1
POWER=$2

# Set clocks
sudo nvidia-smi -ac 1215,1410 -i ${GID}

# Set power
sudo nvidia-smi -pl ${POWER} -i ${GID}

# Log NVIDIA-SMI
nvidia-smi -i ${GID} --query-gpu=timestamp,gpu_name,uuid,persistence_mode,pstate,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sync_boost,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.max.graphics,clocks.max.sm,clocks.max.memory,mig.mode.current --format=csv -f metrics_single_${POWER}.csv --loop-ms=100 &
BACK_PID=$!

declare MIG_MODE=( "0" "5" "9" "14" "19" )
declare GI_IDS=( "0" "2" "1" "3" "9")
declare NUM_MODES=( ${#MIG_MODE[@]} )

######### MIG #########
for (( m=0; m<${NUM_MODES}; m++ ))
do
	echo -e "******************************************************************************************"
	sudo nvidia-smi mig -cgi ${MIG_MODE[${m}]} -i ${GID}
	sudo nvidia-smi mig -cci -gi ${GI_IDS[${m}]} -i ${GID}
	nvidia-smi mig -lgi
	nvidia-smi mig -lci
	MIG_ID="$(nvidia-smi -L | grep -A1 "GPU ${GID}" | grep "MIG" | cut -f3 -d":" | cut -f1 -d"/" | awk '{print $1}')"

	PIDS=()
	for i in "${GI_IDS[${m}]}"
	do
		declare COUNT=(0)
		ID=(`echo $i | sed 's/,/\n/g'`)
		for j in "${ID[@]}"
		do
			CUDA_VISIBLE_DEVICES=${MIG_ID}/${j}/0  ./mig_example ${MIG_MODE[${m}]} ${COUNT} ${MIG_ID}/${j}/0 &
			((COUNT=COUNT+1))
		done
	done

	# -ci 0 because we create single compute instance per gpu instance
	sudo nvidia-smi mig -dci -ci 0 -gi ${GI_IDS[${m}]} -i ${GID}
	sudo nvidia-smi mig -dgi -gi ${GI_IDS[${m}]} -i ${GID}

	sleep 3
	echo -e
done

kill -9 ${BACK_PID}
