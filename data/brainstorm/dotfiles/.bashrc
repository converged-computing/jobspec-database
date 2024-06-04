# Basic

# http://blog.bradlucas.com/posts/2017-11-05-gpg-signing-failed-inappropriate-ioctl-for-device-/
export GPG_TTY=$(tty)

platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'Darwin' ]]; then
   platform='osx'
fi

#export LANG=C.UTF-8
#export LC_ALL=C.UTF-8
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8" 

export PATH="~/bin:$PATH"

# Larger bash history (allow 32³ entries; default is 500)
export HISTSIZE=32768
export HISTFILESIZE=$HISTSIZE
export HISTCONTROL=ignoredups

if [[ "$platform" == 'osx' ]]; then
	alias ls="ls -G"
	alias removexattrs="chmod -RN . && xattr -c ."
	if [ -f $(brew --prefix)/etc/bash_completion ]; then
		source $(brew --prefix)/etc/bash_completion
	fi
else
	alias ls="ls --color"
	if [ -f /etc/bash_completion ]; then
    	source /etc/bash_completion
	fi
fi

# AWS
complete -C aws_completer aws
source $(which assume-role)
# assume-role
export AWS_DEFAULT_PROFILE_ASSUME_ROLE="bastion_brainstorm"
#export AWS_ROLE_SESSION_TIMEOUT="42100"

function aws_account_info {
  [ "$AWS_ACCOUNT_NAME" ] && [ "$AWS_ACCOUNT_ROLE" ] && echo -n "aws:($AWS_ACCOUNT_NAME:$AWS_ACCOUNT_ROLE) "
}

PROMPT_COMMAND='aws_account_info'


# UMCCR
export GITHUB_TOKEN=`cat $HOME/.github`
# export VAULT_ADDR=https://vault.dev.umccr.org:8200
alias assume-role-vault='. /usr/local/bin/assume-role-vault'


# added by travis gem
[ -f /Users/romanvg/.travis/travis.sh ] && source /Users/romanvg/.travis/travis.sh

# linuxbrew
if [[ "$platform" == 'linux' ]]; then
	export PATH="$HOME/.linuxbrew/bin:$PATH"
	export MANPATH="$HOME/.linuxbrew/share/man:$MANPATH"
	export INFOPATH="$HOME/.linuxbrew/share/info:$INFOPATH"
	export CC=${CC:-`which gcc`} && export CXX=${CXX:-`which g++`}
	# Cannot be bothered to pass --env=inherit every time
	function brew {
		~/.linuxbrew/bin/brew "$@" --env=inherit;
	}
fi

# Go
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
export PATH=$PATH:/usr/lib/go-1.7/bin

# Docker & Kubernetes
alias drma="docker ps -aq --no-trunc | xargs docker rm"
alias dkd="docker run -d -P"
alias dki="docker run -t -i -P"
alias dco="docker-compose"
alias dpa="docker ps -a"

source <(kubectl completion bash)
source <(kompose completion bash)

# Espressif toolchains for esp8266 and esp32 and Mongoose-OS
export PATH=$PATH:$HOME/dev/espressif/crosstool-NG/builds/xtensa-esp32-elf/bin:~/.mos/bin
export PATH=$HOME/dev/espressif/esp-open-sdk/xtensa-lx106-elf/bin:$PATH
export IDF_PATH=~/dev/espressif/esp-idf
export ESP_ROOT=~/esp8266/esp-open-sdk
export ESPBAUD=921600
export AMPY_PORT=/dev/cu.usbserial-FTYKHBJT

# Slurm
alias slurm_template='echo "#!/bin/bash

#SBATCH -n 1
#SBATCH -J
#SBATCH -p defq
#SBATCH -o %J.err
#SBATCH -e %J.out
#SBATCH --time=240:00:00
" > slurm.sh'

# KiCad/SKidl
export KISYSMOD="/usr/share/kicad/modules"

# Android
export USE_CCACHE=1
export ANDROID_JACK_VM_ARGS="-Dfile.encoding=UTF-8 -XX:+TieredCompilation -Xmx3G"

# Add RVM to PATH for scripting. Make sure this is the last PATH variable change.
#export PATH="$HOME/.jenv/bin:$PATH"
#eval "$(jenv init -)"

# Python
export PATH=$PATH:$HOME/.local/bin

# Radare

export R2PM_DBDIR="$HOME/.r2pm"
PATH="$PATH:/Users/romanvg/.mos/bin"

# NodeJS

# tabtab source for serverless package
# uninstall by removing these lines or running `tabtab uninstall serverless`
[ -f /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/serverless.bash ] && . /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/serverless.bash
# tabtab source for sls package
# uninstall by removing these lines or running `tabtab uninstall sls`
[ -f /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/sls.bash ] && . /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/sls.bash

# Add RVM to PATH for scripting. Make sure this is the last PATH variable change.
export PATH="$PATH:$HOME/.rvm/bin"
