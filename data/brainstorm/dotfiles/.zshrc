# Basic

alias jsontidy="pbpaste | jq '.' | pbcopy"

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
#	if [ -f $(brew --prefix)/etc/bash_completion ]; then
#		source $(brew --prefix)/etc/bash_completion
	fi
else
	alias ls="ls --color"
	if [ -f /etc/bash_completion ]; then
    	source /etc/bash_completion
	fi
fi

# AWS
alias ssoaws="aws-google-auth -d 43000 -a -I C03v1plyn -S 528518671174 -u roman.valls@umccr.org -p default && export AWS_PROFILE=default"
alias awsdev="pass show vccc/umccr.org | aws-google-auth -d 43000 -r arn:aws:iam::620123204273:role/dev-admin -I C03v1plyn -S 528518671174 -u roman.valls@umccr.org -p default && export AWS_PROFILE=default"
alias awsprod="pass show vccc/umccr.org | aws-google-auth -d 43000 -r arn:aws:iam::472057503814:role/prod-admin -I C03v1plyn -S 528518671174 -u roman.valls@umccr.org -p default && export AWS_PROFILE=default"

# Illumina
alias igplogin="pass show vccc/umccr-beta.login.illumina.com | igp login roman.vallsguimera@unimelb.edu.au -d umccr-beta | cat ~/.igp/.session.yaml | grep access-token | sed -e 's/access-token: //g'"

# GCloud
#source /usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/completion.bash.inc
#source /usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.bash.inc

# UMCCR
export GITHUB_TOKEN=`cat $HOME/.github`
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

# Docker & Kubernetes
#source <(kubectl completion bash)
#source <(kompose completion bash)

alias drma="docker ps -aq --no-trunc | xargs docker rm"
alias dkd="docker run -d -P"
alias dki="docker run -t -i -P"
alias dco="docker-compose"
alias dpa="docker ps -a"

alias killallkube="kubectl delete secret,statefulset,pod,svc,rs,deploy,ingress,secret,configmap --all"
alias watchpods="watch kubectl get pods"
alias watchallpods="watch kubectl get pods --all-namespaces"
alias watchevents="kubectl get -w events"
alias watchsvc="kubectl get -w svc -o wide"
alias killallpods="kubectl delete pods --all"
alias ke="kubectl exec -it"
alias kl="kubectl logs -f"
alias kgs="kubectl get svc -o wide"
alias kgsa="kubectl get svc --all-namespaces -o wide"
alias kgp="kubectl get pod"
alias kgd="kubectl get deployment"
alias kgpa="kubectl get pod --all-namespaces"

# Espressif and Mongoose toolchains for esp8266 and esp32 and Mongoose-OS
export PATH=$PATH:$HOME/dev/espressif/xtensa-esp32-elf/bin:~/.mos/bin
export PATH=$HOME/dev/espressif/esp-open-sdk/xtensa-lx106-elf/bin:$PATH
export IDF_PATH=~/dev/espressif/esp-idf

. ${IDF_PATH}/add_path.sh

# Espressif esp8266
export ESP_ROOT=~/dev/esp8266/esp-open-sdk
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
export KISYSMOD="/Library/Application Support/kicad/modules/"
export KICAD_SYMBOL_DIR="/Library/Application Support/kicad/library"
export KICAD_PTEMPLATES="/Library/Application Support/kicad/template/"

# Android
export USE_CCACHE=1
export ANDROID_JACK_VM_ARGS="-Dfile.encoding=UTF-8 -XX:+TieredCompilation -Xmx3G"

# Add RVM to PATH for scripting. Make sure this is the last PATH variable change.
#export PATH="$HOME/.jenv/bin:$PATH"
#eval "$(jenv init -)"

# Python
export PATH=$PATH:$HOME/.local/bin

# radare & ghidra

export R2PM_DBDIR="$HOME/.r2pm"
export PATH=$PATH:/usr/local/ghidra
alias ghidra="ghidraRun"


# NodeJS

# tabtab source for serverless package
# uninstall by removing these lines or running `tabtab uninstall serverless`
#[ -f /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/serverless.bash ] && . /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/serverless.bash
# tabtab source for sls package
# uninstall by removing these lines or running `tabtab uninstall sls`
#[ -f /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/sls.bash ] && . /usr/local/lib/node_modules/serverless/node_modules/tabtab/.completions/sls.bash

# Add RVM to PATH for scripting. Make sure this is the last PATH variable change.

# ChromeCast

# WIP from https://twitter.com/HackerGiraffe/status/1080085892902125568
#chromecast() {
#  curl -H "Content-Type: application/json" \
#    http://$1:8008/apps/YouTube \
#    -X POST \
#    -d "v=$2";
#}

# Add RVM to PATH for scripting. Make sure this is the last PATH variable change.
export PATH="$PATH:$HOME/.rvm/bin"

# tabtab source for slss package
# uninstall by removing these lines or running `tabtab uninstall slss`
[ -f /Users/romanvg/dev/umccr/htsget-aws/node_modules/tabtab/.completions/slss.bash ] && . /Users/romanvg/dev/umccr/htsget-aws/node_modules/tabtab/.completions/slss.bash
