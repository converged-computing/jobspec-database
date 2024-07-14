# Subshell will source it again because this variable is not exported.
[[ -z ${SOURCE_ONCE_ENV_SH+X} ]] || return
SOURCE_ONCE_ENV_SH=1

set -eu -o pipefail

# USER INPUT: specify the git repo directory and the install directory
BASEDIR="$(pwd)"

# Various directories of note
QIODIR=${BASEDIR}/lqcd/install/qio/
QMPDIR=${BASEDIR}/lqcd/install/qmp/
QUDADIR=${BASEDIR}/quda/build/
EIGENDIR=${BASEDIR}/quda/eigen/Eigen/

GITDIR=$(cd "$(dirname "$BASH_SOURCE")"&&pwd)

# use the same compiler for everything
module load cmake/3.14.5
module load gcc/8.3.1

run(){
	DONE="$BASEDIR/DONE_$1"
	if [[ -f $DONE ]];then
		echo "# SKIPPING $1"
		echo "#	Remove $DONE to force rerun"
		return
	fi
	echo "##########"
	echo "# STARTING $1"
	echo "##########"
	"$GITDIR/$1"
	touch $DONE
}
