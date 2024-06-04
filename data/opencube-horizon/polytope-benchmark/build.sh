#!/usr/bin/env bash
#SBATCH --job-name=build
#SBATCH --qos=nf
#SBATCH --ntasks=32
#SBATCH --mem=32G
#SBATCH --output=build.log

basewd=$PWD

rm -f build.done build.fail

handle_error() {
  set +e
  touch $basewd/build.fail
  trap 0
  exit 1
}
trap handle_error ERR

set -e

prgenvswitchto intel
module load intel/2021.4.0
echo $(module list)

# Build binary packages
mkdir -p build
cd build
if [[ ! -e polytope-bundle ]]; then 
  git clone ssh://git@git.ecmwf.int/~mawj/polytope-bundle.git -v --branch main --single-branch --depth 1 
fi
cd polytope-bundle 
bash polytope-bundle create 
bash polytope-bundle build 
build/install.sh --fast 
cd ../..
BUNDLE_PATH=$(realpath build/polytope-bundle)

export LD_LIBRARY_PATH=$BUNDLE_PATH/install/lib64:$LD_LIBRARY_PATH

module load python3/3.11.8-01 gdal/3.8.4
rm -rf env
python3 -m venv env
source env/bin/activate
python -m pip install -r docker/requirements.txt

deactivate

touch $basewd/build.done