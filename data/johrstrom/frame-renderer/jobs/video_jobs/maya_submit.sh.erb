#!/bin/bash

#set the resources requests. Job name, time, nodes and cores per task, task count 

#PBS -l nodes=1:ppn=<%= cores %>
#PBS -j oe

# TMPDIR currently set to something like /tmp/pbstmp.7431620[1] (unescaped)
# and Maya cannot read filepaths like that correctly.
JOB_TMPDIR=$TMPDIR
BIND_DIRS="$JOB_TMPDIR:/tmp"
# set a new tmpdir bc that's what the container needs to look at
export TMPDIR="/tmp"

UTIL_FILE="$JOB_TMPDIR/utils.rb"

cat > "$UTIL_FILE" <<UTILS
#!/usr/bin/env ruby

def create_thumbnails
  img_dir="<%= project_dir %>/images"
  thumb_dir="<%= project_dir %>/thumbnails"
  task_start_frame = (ENV['TASK_START_FRAME'] || 1).to_i
  task_end_frame = (ENV['TASK_END_FRAME'] || 1).to_i

  images = Dir.glob("#{img_dir}/*").keep_if do |img|
    file_num = /\d+/.match(File.basename(img))
    if !file_num.nil?
      num = file_num.to_s.to_i
      num >= task_start_frame && num <= task_end_frame
    else
      false
    end
  end

  convert_args="-resize 80x80"
  Dir.mkdir(thumb_dir) unless File.exists?(thumb_dir)

  images.each do |img|
    fname = File.basename(img, ".*")
    puts "creating thumbnail for #{fname}"
    system("convert #{img} #{convert_args} #{thumb_dir}/#{fname}.png")
  end

end
UTILS

echo "starting at $(date)"

module purge
#load the singularity and ACCAD modules and record the loaded module list
module use /usr/local/share/lmodfiles/project/osc
module load singularity/current
module load ruby

<%
  groups = OodSupport::Process.groups.map(&:name)
%>

<% if groups.include?('mayakentst') %>
module load project/kent
module load maya/2020-kent
<% else %>
<%# this module blocked by group membership on the file system, so OK to be in else block %>
module load project/accad
module load maya/2020-accad
<% end %>

mkdir -p $JOB_TMPDIR/Autodesk
BIND_DIRS="$BIND_DIRS,$JOB_TMPDIR/Autodesk:/var/opt/Autodesk"

# init the tmpdir
TMPDIR=$JOB_TMPDIR maya_init.sh

# set all the different variables you'll need 
CMD="singularity exec -B $BIND_DIRS $MAYA_IMG Render"
RENDERER="<%= renderer %>"
EXTRA_ARGS="<%= extra %>"
FILE="<%= file %>"
PRJ_DIR="<%= project_dir %>"
SKIP_EXISTING="-skipExistingFrames <%= skip_existing %>"

# added 'shift' to this array so we don't have to deal with PBS_ARRAYID-1
# calculations, we can just shift right +1
START_FRAMES=(shift <%= task_start_frames.join(' ')%>)
END_FRAMES=(shift <%= task_end_frames.join(' ')%>)

export TASK_START_FRAME=${START_FRAMES[$PBS_ARRAYID]}
export TASK_END_FRAME=${END_FRAMES[$PBS_ARRAYID]}

ARGS="-r $RENDERER $SKIP_EXISTING ${EXTRA_ARGS} -proj $PRJ_DIR -s $TASK_START_FRAME -e $TASK_END_FRAME $FILE"

echo "executing: $CMD $ARGS"
echo "on host: $(hostname)"
echo "with modules:"
module list

# now let's actually execute and grab the end status
${CMD} ${ARGS}
STATUS=$?

echo "ended at $(date)"
echo "ended with status $STATUS"

if [ $STATUS -eq 0 ]; then
  echo "now creating thumbnails"
  ruby -r $UTIL_FILE -e create_thumbnails
fi

exit $STATUS
