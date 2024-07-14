export PATH=/usr/local/nvidia:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia:${LD_LIBRARY_PATH}

source /etc/larbys.sh
cd /usr/local/larbys/ssnet_example/sw
source setup.sh
cd /cluster/kappa/90-days-archive/wongjiradlab/grid_jobs/ssnet-tuftscluster-scripts