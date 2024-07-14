#! /bin/bash -l

#SBATCH -J bs2-run
#SBATCH -t 48:00:00
#SBATCH -p core
#SBATCH -n 1
#SBATCH -M rackham
#SBATCH --output=bs2-run.err

# Slurm script for birdscanner2-run
#
# Test by using:
#     sbatch -A <accountnr> --test-only workflow/scripts/bs2-run.slurm.sh
#
# Start by using:
#     sbatch -A <accountnr> workflow/scripts/bs2-run.slurm.sh
#
# Stop by using:
#     scancel 1234
#     scancel -i -u $USER
#     scancel --state=pending -u $USER
#
# Monitor by using:
#    jobinfo -u $USER -M rackham
#    sinfo -p devel
#    squeue
#

module load bioinfo-tools
module load snakemake/5.10.0
module load hmmer/3.2.1-intel
module load blast/2.9.0+

make slurm-run

>&2 echo ""
>&2 echo ""
>&2 echo "bs2-run job should now have been submitted to cluster."
>&2 echo "Submission details, and any possible errors, are in the bs2-run.err file."
>&2 echo "Monitor submitted jobs with with the 'jobinfo' command."
>&2 echo "When all processing are finished, you should see outfiles"
>&2 echo "in folder birdscanner2/results."
>&2 echo ""
>&2 echo "Reached the end of the bs2-run slurm script."
