#!/bin/bash


#10/10/2021 M. Farman (farman@uky.edu)

#SBATCH --mail-type ALL                         #Send email on start/end
#SBATCH --mail-user farman@uky.edu            #Where to send email
#SBATCH --account=gol_farman_uksr              #Name of account to run under
#SBATCH --partition=V4V32_SKY32M192_L           #Partition

#SBATCH --job-name=dorado-gpu    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=28G                # total memory per node (4 GB per cpu-core is defau$
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)

pod5s=$1

container=/share/singularity/images/ccs/conda/lcc-conda-8-rocky8.sinf

module load ccs/singularity-3.8.2

singularity run --nv --app dorado034 $container dorado download --model dna_r9.4.1_e8_hac@v3.3

singularity run --nv --app dorado034 $container dorado basecaller --device 'cuda:all' dna_r9.4.1_e8_hac@v3.3 --emit-fastq $pod5s > ${pod5s/pod5_/}.fastq
