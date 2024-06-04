#!/bin/bash
bfile="$1"
njobs="$2"
cuda="$3"
if [ -z "$bfile" ] || [ -z "$njobs" ]; then
	echo "Usage: $0 <file.blend> <number of jobs> [force cuda:Y/n]"
	exit 1
fi

# get full blender path
bfile="$(readlink -f "$bfile")"
blender="$(which blender)"
bash="$(which bash)"
ffmpeg="$(which ffmpeg)"

# make directories
scratch="/scratch3/$USER"
outdir="$scratch/$(basename ${bfile/.blend/})"
vfile="$scratch/$(basename ${bfile/.blend/.mp4})"
mkdir -p "$outdir"

# declare simplified qsubs
queue="qsub -j oe -o $outdir/log"
rargs="-l select=${njobs}:ncpus=16:ngpus=2:mem=32gb,walltime=48:00:00"
bargs="--enable-autoexec --background -t 16"

if [ -z "$cuda" ] || [ "$cuda" = "Y" ] || [ "$cuda" = "y" ]; then
	cargs="--python-expr '\"import bpy; bpy.context.user_preferences.addons[\\\"cycles\\\"].preferences.compute_device_type = \\\"CUDA\\\"; bpy.context.scene.cycles.device = \\\"GPU\\\"\"'"
else
	cargs=""
fi

run() {
	# queue up a generated PBS file
	$queue <<EOF
#PBS -N blender_$(basename ${bfile/.blend/})
#PBS $rargs

module add gnu-parallel

# get all frames from Blender API
total="\$('$blender' $bargs '$bfile' --python-expr 'import bpy; print(bpy.context.scene.frame_end)' 2>/dev/null | grep '^[[:digit:]]' | head -n1)"

# truncate frames to render file
echo -n >"$outdir/frames"

# add all frames that have not been rendered
for frame in \$(seq \$total); do
	# check if the specified output file has been rendered and add it if not
	[ -f "\$(printf "$outdir/%04d.png" \$frame)" ] || echo "\$frame" >>"$outdir/frames"
done

# get a total number of frames to render
frames=\$(wc -w <"$outdir/frames")

# bail if there are no frames to render
[ \$frames -gt 0 ] || exit

# split frames to render into $njobs parts of roughly equal size with Python and send that with a blender command to parallel
python <<IEOF | xargs -I'{}' echo '"$blender"' $bargs '"$bfile"' $cargs --render-output '"$outdir/"' --render-format PNG --render-frame '{}' | parallel --sshloginfile "\$PBS_NODEFILE" -j 1
from __future__ import division, print_function

# read all of the lines in the frames file
with open('$outdir/frames', 'r') as framefile:
	frames = framefile.read().splitlines()

# calculate avg number of frames per job
avg = len(frames)/$njobs
last = 0.0

# keep going until we are out of frames
while last < len(frames):
	# if this chunk has frames in it
	if int(last) != int(last + avg):
		# comma separate a list of frames roughly the size of avg
		print(','.join(frames[int(last):int(last + avg)]))
	# remember added frames
	last += avg
IEOF

# stich together frames into a video file
"$ffmpeg" -i "$outdir/%04d.png" -framerate 30 -pix_fmt yuv420p -vcodec libx264 "$vfile"
EOF
}

jobid="$(run)"

echo "Job ID: $jobid"

# tell user
echo "Rendering '$vfile' (ctrl+c to retrieve video file later)..."

trap "echo && echo \"Video file will be at '$vfile' after job '$jobid' finishes.\" && exit" SIGINT SIGTERM

# wait on ffmpeg
while qstat $jobid >/dev/null; do
	sleep 1
done

# tell user
echo "Video available at '$vfile'."
