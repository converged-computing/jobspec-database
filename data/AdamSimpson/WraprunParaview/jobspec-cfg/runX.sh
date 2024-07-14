#!/bin/bash

# Store and then unset LD_PRELOAD
OLD_PRELOAD=$LD_PRELOAD
unset LD_PRELOAD

# Launch X and VNC as normal
startx &
sleep 5
starttvnc :1 &
export DISPLAY=:1

# Set LD_PRELOAD to the original LD_PRELOAD and add in wraprun required preloads
LD_PRELOAD=$OLD_PRELOAD:$WRAPRUN_PRELOAD
echo $LD_PRELOAD

# Load Paraview
module load paraview

# Launch pvbatch
vglrun pvbatch "$@"
