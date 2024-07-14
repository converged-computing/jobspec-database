#!/bin/bash
echo "Hello, World!"

module load comsol matlab
comsol server -np 2 &
matlab -batch "DesignSpaceStudy1D('test','Wm+We',1)" 

