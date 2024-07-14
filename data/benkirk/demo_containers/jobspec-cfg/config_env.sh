# if [ -x /glade/u/home/benkirk/bugreports/spack/charliecloud/spack/share/spack/setup-env.sh ]; then
#     . /glade/u/home/benkirk/bugreports/spack/charliecloud/spack/share/spack/setup-env.sh || exit 1
#     spack env activate -p container_env || exit 1
# else
#     . ~benkirk/spack_modules.sh && module load podman charliecloud singularityce || exit 1
#     spack env activate -p container_env || exit 1
# fi

module --force purge && module load ncarenv/23.05 && module load charliecloud podman apptainer && module list

# preliminaries - podman at least seems to require a local filesystem, try leaving TMPDIR on lustre
# and I see failures...
TMPDIR=/var/tmp/${USER}/container_tmp && mkdir -p ${TMPDIR}
clean_container_dirs()
{
    chmod -R u+rwX /var/tmp/${USER}*
    rm -rf /var/tmp/${USER}*
    mkdir -p /var/tmp/${USER}
    [ -d ${TMPDIR} ] || mkdir -p ${TMPDIR}
}

for exe in podman ch-image singularity ; do
    which $exe && $exe --version && echo || exit 1
done


gotocolumn="\033["$rescol"G";
white="\033[01;37m";
green="\033[01;32m";
red="\033[01;31m";
grey="\033[00;37m";
cyan="\033[01;36m";
colorreset="\033[m";


# Write pretty status message
function message_running {
    echo " "
    echo -e $cyan"-------------------------------------------------------------------------------------"
    echo -e $cyan'(test): Running' $@
    echo -e $cyan"-------------------------------------------------------------------------------------"
    echo -e -n $colorreset;
}

# Write pretty status message
function message_cmd {
    echo -e $cyan'(test): Running' $@
    echo -e -n $colorreset;
    eval $@
}

# Write failure message
function message_failed {
    echo -e $red"-------------------------------------------------------------------------------------"
    echo -e $red'(test): ** ERROR:' $@
    echo -e $red"-------------------------------------------------------------------------------------"
    echo -e -n $colorreset;
    #exit 1
}

# Write pretty pass message
function message_passed {
    echo -e $green"-------------------------------------------------------------------------------------"
    echo -e $green'(test): PASSED:' $@
    echo -e $green"-------------------------------------------------------------------------------------"
    echo -e -n $colorreset;
}


function try_command {
    message_cmd $@ \
	&& message_passed $@ \
	|| message_failed $@
}
