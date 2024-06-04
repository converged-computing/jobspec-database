#!/bin/bash
#SBATCH --job-name=01112019_2.5sd_vid_gen
#SBATCH --output=cluster/01112019_2.5sd_vid_gen.out
#SBATCH --error=cluster/01112019_2.5sd_vid_gen.err
#SBATCH --partition=psych_week
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=FAIL

module load MATLAB/2022b

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/gpfs/milgram/project/chang/pg496/repositories/categorical/lib/linux"

matlab -r "addpath( genpath( '/gpfs/milgram/project/chang/pg496/repositories' ) ); session='01112019'; params=sg_disp.util.get_params_for_cluster(); sg_disp.cluster.generate_video_for_one_session( session, params );"
