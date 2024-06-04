# List pbs log files .e and .o files with 6 digits after them
lseo() {
    local search_dir="."
    local search_string=".*\.[eo][0-9]{6}$"
    local delete=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) echo "Usage: lseo [-e|-o] [-delete] [search_dir]"; return 0;;
            -e) search_string=".*\.e[0-9]{6}$"; shift;;
            -o) search_string=".*\.o[0-9]{6}$"; shift;;
            -delete) delete="-delete"; shift;;
            *) search_dir="$1"; shift;;
        esac
    done

    find "$search_dir" -maxdepth 1 -type f -regextype posix-extended -regex "$search_string" $delete
}

# Make rmeo same as lseo -delete
alias rmeo="lseo -delete"

# Change directory to bioinformatics database directory
alias cddb="cd /projects/bioinformatics/DB/"

# Push database directory
alias pushdb="pushd /projects/bioinformatics/DB/"

# Delete all jobs belonging to the current user
alias qdelall="qselect -u $USER | xargs qdel"

# Watch job status continuously for the current user
alias watchjobs="watch -n 1 qstat -fu $USER"
alias wj="watch -n 1 qstat -fu $USER"  # Shortcut alias

# Search for 'Killed' messages in files
alias anykilled="find . -type f -exec grep -l 'Killed' {} +"
#alias anykilled="grep -l 'Killed' ./*"

# Function to create and submit a PBS job
fsub() {
    local script_path="./pbs_job.sh"  # Path to save the PBS script
    local job_name="pbs_job"     # Default name of the job
    local cddir="."     # cd to this directory before running the command
    local num_cpus="1"     # Default number of CPUs
    local num_gpus="0"     # Default number of GPUs
    local memory="1gb"       # Default memory allocation
    local queue="q02anacreon"        # Default queue name
    local walltime=""     # Default walltime
    local environment="base"     # Default conda environment
    local command="echo \"No command was provided!\""      # Command to run
    local waitinglist=""     # Job IDs to wait for, accepted sepataors: space, comma, semicolon, colon

    # Parse command line options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p) script_path="$2"; shift 2;;
            -n) job_name="$2"; shift 2;;
            -cd) cddir="$2"; shift 2;;
            -nc) num_cpus="$2"; shift 2;;
            -ng) num_gpus="$2"; shift 2;;
            -m) memory="$2"; shift 2;;
            -q) queue="$2"; shift 2;;
            -wt) walltime="$2"; shift 2;;
            -e) environment="$2"; shift 2;;
            -c) command="$2"; shift 2;;
            -w) waitinglist="$2"; shift 2;;
            -h|--help) echo "Usage: fastsub [-p script_path] [-n job_name] [-cd cddir] [-nc num_cpus] [-ng num_gpus] [-m memory] [-q queue] [-wt walltime] [-e environment] [-c command] [-w waitinglist]"; return 0;;
            *) echo "Unknown option: $1"; return 1;;
        esac
    done

    # If both job_name and script_path are default, try to infer them from the command
    if [[ "$job_name" == "pbs_job" && "$script_path" == "./pbs_job.sh" ]]; then 
        # check if command is python or R script
        if [[ "$command" == *"python"* ]]; then
            # find word ending with .py and replace .py with .sh
            script_path=$(echo "$command" | grep -o '\w*\.py' | sed 's/\.py/.sh/')
            job_name=$(basename "$script_path")
            job_name="${job_name%.*}"
        elif [[ "$command" == *"Rscript"* ]]; then
            # find word ending with .R and replace .R with .sh
            script_path=$(echo "$command" | grep -o '\w*\.R' | sed 's/\.R/.sh/')
            job_name=$(basename "$script_path")
            job_name="${job_name%.*}"
        fi
    fi


    # If script_path does not end with .sh, append .sh to it
    if [[ ! "$script_path" == *.sh ]]; then
        script_path="$script_path.sh"
    fi
    # If job name ends with .sh, remove it
    if [[ "$job_name" == *.sh ]]; then
        job_name="${job_name%.*}"
    fi

    # If job_name is default, but script_path is not, use the script name as job_name and vice versa
    if [[ "$job_name" == "pbs_job" && "$script_path" != "./pbs_job.sh" ]]; then
        job_name=$(basename "$script_path")
        job_name="${job_name%.*}"
    fi
    if [[ "$script_path" == "./pbs_job.sh" && "$job_name" != "pbs_job" ]]; then
        script_path="./$job_name.sh"
    fi

    # If ngpus in not 0 and queue is default, change queue to q02gaia
    if [[ "$num_gpus" -gt 0 && "$queue" == "q02anacreon" ]]; then
        queue="q02gaia"
    fi

    # get absolute path of cddir
    cddir=$(realpath "$cddir")

    # Create the PBS script
    touch "$script_path"
    echo "#!/bin/bash" > "$script_path"
    echo "#PBS -N $job_name" >> "$script_path"
    echo "#PBS -l select=1:ncpus=$num_cpus:ngpus=$num_gpus:mem=$memory" >> "$script_path"
    echo "#PBS -q $queue" >> "$script_path"
    if [[ -n "$walltime" ]]; then
        echo "#PBS -l walltime=$walltime" >> "$script_path"
    fi

    # Change directory to cddir, create it if it does not exist
    echo "mkdir -p $cddir" >> "$script_path"
    echo "cd $cddir" >> "$script_path"

    # Activate conda environment
    echo 'eval "$(/cluster/shared/software/miniconda3/bin/conda shell.bash hook)"' >> "$script_path"
    echo "conda activate $environment" >> "$script_path"

    # Add command to script
    echo "$command" >> "$script_path"

    echo "exit 0" >> "$script_path"
    
    # If waiting list is not a set of strings separated by colons,
    # then substitute all spaces, commas, and semicolons with colons 
    # and substitute all multiple colons with a single colon
    if [[ -n "$waitinglist" ]]; then
        waitinglist=$(echo "$waitinglist" | tr ' ,;' ':')
        waitinglist=$(echo "$waitinglist" | tr -s ':')
        waitinglist=$(echo "$waitinglist" | sed 's/^:\(.*\)$/\1/')
    fi

    # Submit the job
    if [[ -n "$waitinglist" ]]; then
        job_id=$(qsub -W depend=afterok:"$waitinglist" "$script_path")
    else
        job_id=$(qsub "$script_path")
    fi

    echo $job_id
}

findjobs() {
    local search_string=""
    local case_insensitive=false
    local search_queue=false
    local delete=false

    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                echo "Usage: findjobs [-i] [-q] [-d] [search_string]"
                echo "  -i: case insensitive search"
                echo "  -q: search in the queue name instead of job name"
                echo "  -delete: delete the jobs found"
                echo "  search_string: search string to look for in job names or queue"
                return 0
                ;;
            -i)
                case_insensitive=true
                shift
                ;;
            -q)
                search_queue=true
                shift
                ;;
            -delete)
                delete=true
                shift
                ;;
            *)
                search_string="$1"
                shift
                ;;
        esac
    done

    # Extract job IDs based on the search string in job names or queue
    if [ -z "$search_string" ]; then
        # If no search string provided, return all jobs
        if [ "$search_queue" = true ]; then
            local job_ids=$(qstat -u $USER | awk 'NR>5 {print $1}')
        else
            local job_ids=$(qstat -u $USER | awk 'NR>5 {print $1}')
        fi
    else
        # Search for job IDs based on the provided search string
        if [ "$case_insensitive" = true ]; then
            if [ "$search_queue" = true ]; then
                local job_ids=$(qstat -u $USER | awk -v search="$(echo "$search_string" | tr '[:upper:]' '[:lower:]')" 'NR>5 && tolower($3) ~ search {print $1}')
            else
                local job_ids=$(qstat -u $USER | awk -v search="$(echo "$search_string" | tr '[:upper:]' '[:lower:]')" 'NR>5 && tolower($4) ~ search {print $1}')
            fi
        else
            if [ "$search_queue" = true ]; then
                local job_ids=$(qstat -u $USER | awk -v search="$search_string" 'NR>5 && $3 ~ search {print $1}')
            else
                local job_ids=$(qstat -u $USER | awk -v search="$search_string" 'NR>5 && $4 ~ search {print $1}')
            fi
        fi
    fi
    
    # Check if any jobs found
    if [ -z "$job_ids" ]; then
        return 1
    fi

    # Delete the jobs if -delete option is provided
    if [ "$delete" = true ]; then
        echo "Deleting jobs with the search string '$search_string':"
        echo "$job_ids"
        echo "$job_ids" | xargs qdel
    else
        echo "$job_ids"
    fi
}

# Alias for findjobs
alias fj="findjobs"

mkenv () {
    local name=""
    local type=""
    local version=""
    local jupyter=""
    local pip_packages=""
    local conda_packages=""
    local ask="true"

    # parse arguments, name is required
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -n|--name)
                name="$2"
                shift
                shift
                ;;
            -t|--type)
                type="$2"
                shift
                shift
                ;;
            -v|--version)
                version="$2"
                shift
                shift
                ;;
            --jupyter)
                jupyter="$2"
                shift
                shift
                ;;
            --pip)
                pip_packages+=" $2"
                shift
                shift
                ;;
            --conda)
                conda_packages+=" $2"
                shift
                shift
                ;;
            --noask)
                ask=false
                shift
                ;;
            -h|--help)
                echo "Usage: mkenv -n <name> [-t <type>] [-v <version>] [--jupyter <true|false>] [--pip <package1 padkage2 ...>] [--conda <package1 padkage2 ...>] [--noask] [-h]"
                echo "  -n, --name: name of the virtual environment"
                echo "  -t, --type: type of the virtual environment ('python', 'R' or 'rpy2')"
                echo "  -v, --version: python version (only for python or rpy2 virtual environments)"
                echo "  --jupyter: whether to install jupyter and expose the kernel, default is true"
                echo "  --pip: list of pip packages to install"
                echo "  --conda: list of conda packages to install"
                echo "  --noask: don't ask for confirmation (use with caution)"
                echo "  -h, --help: show this help message"
                return 0
                ;;
            *)
                # unknown option
                echo "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # set default values if not provided by user
    type=${type:-'python'}
    version=${version:-'3.10'}
    jupyter=${jupyter:-'true'}
    pip_packages=${pip_packages:-'numpy pandas scipy matplotlib'}
    conda_packages=${conda_packages:-''}
    ask=${ask:-'true'}

    # check if name is set
    # if not, exit with error
    if [ -z "$name" ]; then
        echo "Error: name is required"
        return 1
    fi


    # check if type is valid if not, exit with error
    if [ "$type" != "python" ] && [ "$type" != "R" ] && [ "$type" != "rpy2" ]; then
        echo "Error: type must be 'python', 'R' or 'rpy2'"
        return 1
    fi

    # for some reason, you need to source the scripts even if you have already run conda init
    # see: https://github.com/conda/conda/issues/7980#issuecomment-441358406
    source /cluster/shared/software/miniconda3/etc/profile.d/conda.sh
    source /cluster/shared/software/miniconda3/etc/profile.d/mamba.sh

    # check if env with the same name already exists in ~/.conda/envs/
    # if so, ask if user wants to overwrite if yes, remove existing env
    if [ -d "$HOME/.conda/envs/$name" ]; then
        echo "Warning: conda env with name '$name' already exists"
        if [ "$ask" == "true" ]; then
            read -p "Do you want to overwrite it? [y/N] "
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo
                echo "Removing existing env '$name'"
                mamba deactivate
                mamba env remove -n $name
            else
                echo
                echo "Exiting"
                return 0
            fi
        elif [ "$ask" == "false" ]; then
            echo "Removing existing env '$name'"
            mamba deactivate
            mamba env remove -n $name
        fi
    fi

    # check if jupyter kernel with the same name already exists in ~/.local/share/jupyter/kernels/
    # if so, ask if user wants to overwrite
    # default is no
    # if yes, remove existing kernel
    if [ -d "$HOME/.local/share/jupyter/kernels/$name" ]; then
        echo "Warning: jupyter kernel with name '$name' already exists"
        if [ "$ask" == "true" ]; then
            read -p "Do you want to overwrite it? [y/N] "
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo
                echo "Removing existing kernel '$name'"
                rm -rf $HOME/.local/share/jupyter/kernels/$name
            else
                echo
                echo "Exiting"
                return 0
            fi
        elif [ "$ask" == "true" ]; then
            echo "Removing existing kernel '$name'"
            rm -rf $HOME/.local/share/jupyter/kernels/$name
        fi
    fi

    # print message recapping the arguments
    echo "Creating virtual environment with the following arguments:"
    echo "  name: $name"
    echo "  type: $type"
    echo "  version: $version"
    echo "  jupyter: $jupyter"
    echo "  pip_packages: $pip_packages"
    echo "  conda_packages: $conda_packages"
    echo ""

    # if type is R or rpy2, add r-base to the list of conda_packages
    if [ "$type" == "R" ] || [ "$type" == "rpy2" ]; then
        conda_packages="$conda_packages r-base"
        # if jupyter is true, add r-essentials, xorg-libx11, xorg-libxext, xorg-libxrender and xorg-libxt
        if [ "$jupyter" == "true" ]; then
            conda_packages="$conda_packages r-essentials xorg-libx11 xorg-libxext xorg-libxrender xorg-libxt"
        fi
    fi

    # if type is python or rpy2, add python to the list of conda_packages
    if [ "$type" == "python" ] || [ "$type" == "rpy2" ]; then
        conda_packages="$conda_packages python=$version"
        # if jupyter is true, add ipykernel, ipywidgets and jupyterlab_widgets
        if [ "$jupyter" == "true" ]; then
            conda_packages="$conda_packages ipykernel ipywidgets jupyterlab_widgets"
        fi
    fi

    # if type is rpy2, add rpy2 to the list of conda_packages
    if [ "$type" == "rpy2" ]; then
        conda_packages="$conda_packages rpy2"
    fi

    # if type is python, list the conda_packages that will be installed
    echo "Packages that will be installed:"
    echo "conda: $conda_packages"
    if ([ "$type" == "python" ] || [ "$type" == "rpy2" ]) && [ "$pip_packages" != "" ]; then
        echo "pip: $pip_packages"
    fi

    
    # ask for confirmation
    # default is no
    if [ "$ask" == "true" ]; then
        read -p "Do you want to continue? [y/N] "
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo
            echo "Exiting"
            return 0
        fi
    fi

    # deactivate any existing virtual environment
    mamba activate base

    # create virtual environment
    if [ "$ask" == "true" ]; then
        mamba create -n $name $conda_packages
    else
        mamba create -n $name $conda_packages --yes
    fi

    # activate new virtual environment
    mamba activate $name

    # if type python or rpy and pip packages not empty, install pip packages
    if ([ "$type" == "python" ] || [ "$type" == "rpy2" ]) && [ "$pip_packages" != "" ]; then
        pip install $pip_packages
    fi

    # Expose the kernel: python or rpy2
    if [ "$type" == "python" ] || [ "$type" == "rpy2" ]; then
        python -m ipykernel install --user --name $name --display-name $name
        # if type is rpy2 set the R_HOME environment variable in the kernel json to ~/.conda/envs/$name/lib/R
        if [ "$type" == "rpy2" ]; then
            file_path="$HOME/.local/share/jupyter/kernels/$name/kernel.json"
            destination="$HOME/.conda/envs/$name/lib/R"
            # check if file exists
            if [ -f "$file_path" ]; then
                # Backup the original file
                cp "$file_path" "${file_path}.bak"
                # Use sed to update the JSON file
                sed -i '/"metadata": {/a \ \ "env": {\n \ \ \ "R_HOME": "'"$destination"'"\n \ \ },' "$file_path"
                echo "file '$file_path' updated with R_HOME=$destination"
                echo "original file backed up to '${file_path}.bak'"
            else
                echo "Error: file '$file_path' not found"
                return 1
            fi
        fi
    fi

    # Expose the kernel: R
    if [ "$type" == "R" ]; then
        conda config --add channels conda-forge
        conda config --set channel_priority strict
        
        # find the current installation of JupyterLab, it is located in /cluster/shared/software/ and
        # follows the pattern jupyterlab_yymmdd, so it starts with jupyterlab_ and ends with 6 digits
        
        jupyterlab_path=$(ls -l /cluster/shared/software/ | grep "jupyter_[0-9]\{8\}$" | awk '{print $9}')
        jupyterlab_path="/cluster/shared/software/$jupyterlab_path"

        echo "Located JupyterLab installation at $jupyterlab_path"

        # Install IRkernel package
        echo "Installing IRkernel"
        Rscript -e "install.packages('IRkernel', repos='http://cran.r-project.org')"

        # prepend jupyterlab_path to PATH
        new_path="$jupyterlab_path/bin:$PATH"

        # expose kernel from R shell starting it with the new PATH
        echo "Exposing kernel"
        PATH="$new_path" Rscript -e "IRkernel::installspec(name = '$name', displayname = '$name')"
    fi
}

resub () {
    local ask="true"
    local remove_eo="true"

    # parse arguments
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            --noask)
                ask=false
                shift
                ;;
            --keep)
                remove_eo=false
                shift
                ;;
            -h|--help)
                echo "Resubmit the last qsub or fsub command in history"
                echo "Usage: resub [--noask] [--keep]"
                echo "  --noask: don't ask for confirmation (use with caution)"
                echo "  --keep: don't remove .e and .o files from current directory"
                return 0
                ;;
            # Unknown option
            *)
                echo "Unknown option: $1"
                return 1
                ;;
        esac
    done

    # find last run qsub or fsub command and copy it
    last_command=$(history | grep -E 'qsub|fsub' | tail -n 1 | sed 's/^.\{7\}//')
    # if no command found, exit with error
    if [ -z "$last_command" ]; then
        echo "Error: no qsub or fsub command found in history"
        return 1
    else
        echo "Last command:"
        echo "$last_command"
    fi

    if [ "$ask" == "true" ]; then
        read -p "Do you want to resubmit? [Y/n] "
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo "Exiting"
            return 0
        fi
    fi 

    # Resubmit the job
    eval "$last_command"

    # Remove log files from current directory
    if [ "$remove_eo" == "true" ]; then
        if [[ "$ask" == "true" ]]; then
            # Ask, but default to yes
            read -p "Do you want to remove log files from current directory? [Y/n] "
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                rmeo
            fi
        else
            echo "Removing log files from current directory"
            rmeo
        fi
    fi

    # If current directory contains a subdirectory containing only log files, empty it
    if [ "$remove_eo" == "true" ]; then
        for dir in $(find . -mindepth 1 -maxdepth 1 -type d); do
            # use lseo to get list of log files in the directory and compare with the list of all files
            # if the lists are the same, run rmeo in the directory
            if [ "$(lseo $dir)" == "$(find $dir -maxdepth 1 -type f)" ]; then
                echo "Found directory containing only log files: $dir"
                if [ "$ask" == "true" ]; then
                    # Ask, but default to yes
                    read -p "Do you want to empty the directory? [Y/n] "
                    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                        rmeo $dir
                    fi
                else
                    echo "Emptying directory: $dir" 
                    rmeo $dir
                fi
            fi
        done
    fi

}
