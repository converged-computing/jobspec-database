
#! /bin/bash

export DEEPLEPTON=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$DEEPLEPTON

cd $DEEPLEPTON
export PYTHONPATH=$DEEPLEPTON/modules:$PYTHONPATH
export PYTHONPATH=$DEEPLEPTON/modules/datastructures:$PYTHONPATH
export PATH=$DEEPLEPTON/scripts:$PATH

export LD_LIBRARY_PATH=$DEEPLEPTON/modules/compiled:$LD_LIBRARY_PATH
export PYTHONPATH=$DEEPLEPTON/modules/compiled:$PYTHONPATH

