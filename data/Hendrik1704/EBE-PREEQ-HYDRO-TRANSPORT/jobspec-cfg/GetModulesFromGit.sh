#!/bin/bash

echo "Download KoMPoST from GitHub:"
git clone --depth 1 https://github.com/Hendrik1704/KoMPoST.git
cd KoMPoST
git checkout 2f2d65692f5a6f4c0fe99072f92bb6587b7aa2c4
cd ..

echo "Download MUSIC from GitHub:"
git clone --depth 1 https://github.com/Hendrik1704/MUSIC
cd MUSIC
git checkout 13d5cb64c3a86f98bb967bfc46fb28cd4a20713a
cd ..

echo "Download iSS from GitHub:"
git clone --depth 1 https://github.com/chunshen1987/iSS.git
cd iSS
git checkout 7d39d84ff95925bf3bc0edfaf8aae1ac5a28b387
cd ..

echo "Download SMASH from GitHub:"
git clone --depth 1 https://github.com/smash-transport/smash.git --branch SMASH-3.1

echo "Download Pythia (used for SMASH):"
wget https://pythia.org/download/pythia83/pythia8310.tgz

echo "Download stable Eigen library version 3.4.0:"
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz

echo "Downloaded all the modules successfully"
