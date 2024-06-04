#!/bin/sh

# Job name
#PBS -N news_tls

# Output file
#PBS -o news_tls_output.log

# Error file
#PBS -e news_tls_err.log

# request resources and set limits
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:ngpus=4:mem=16GB
#:ompthreads=24
# 'select' chooses number of nodes.

#  load required modules
module load lang/python/anaconda/ lang/cuda
source activate news_tls

# We might need to add the global paths to our code to the pythonpath. Also set the data directories globally.
cd /home/hs20307/news-tls
export PYTHONPATH=$PYTHONPATH:"/home/hs20307/news-tls/"

#  run the script
export DATASET=/work/hs20307/Dataset
export RESULT=./result
export TOKENIZERS_PARALLELISM=false
export PERL5LIB="/home/hs20307/perl5/lib/perl5"

python -u ./experiments/evaluate.py --dataset $DATASET/t17 --method clust --output $RESULT/t17.clust.json

# To submit: qsub run_NER_EMNLP19.sh
# To display the queue: qstat -Q gpu (this is usually where the GPU job ends up)
# Display server status: qstat -B <server>
# Display job information: qstat <jobID>

# To monitor job progress:
# qstat -f | grep exec_host
# Find the node where this job is running.
# ssh to the node.
# tail /var/spool/pbs/spool/<job ID>.bp1.OU
