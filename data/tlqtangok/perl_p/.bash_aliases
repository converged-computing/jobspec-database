alias ssh_arctest4='ssh -X linqi@arctest4'
alias ssh_arcdev4='ssh -X linqi@arcdev4'
alias ssh_='expect $perl_p/ssh_arcdev4.exp ' 
alias ssh_53='expect $perl_p/ssh_.exp tanglinqi 172.16.56.53 904 22'
alias ssh_32='expect $perl_p/ssh_.exp tanglinqi 172.16.56.32 904 22'
alias ssh_bgi='expect $perl_p/ssh_.exp bgi902 172.16.56.32 qwer1234 22'
#alias ssh_19_bgi='expect $perl_p/ssh_.exp bgi902 172.16.56.2 qwer1234 22'
alias ssh_19_tanglinqi='expect $perl_p/ssh_.exp tanglinqi 172.16.56.2 123456 22'
alias ssh_4_bgi='expect $perl_p/ssh_.exp bgi902 172.16.29.10 qwer1234p100 22'
alias ssh_45_bgi='expect $perl_p/ssh_.exp bgi902 172.16.29.7 qwer1234 22'
alias ssh_15_bgi='expect $perl_p/ssh_.exp phoenix 10.227.2.15 qwer1234 22'
alias ssh_ls='expect $perl_p/ssh_.exp ls 172.16.64.21 ls 22'
alias ssh_ddu='expect $perl_p/ssh_.exp root 39.108.3.14 pw 22'
alias ssh_algo='expect $perl_p/ssh_.exp root 39.108.3.14 pw 22'
alias ssh_ddu_jd='expect $perl_p/ssh_.exp jd 39.108.3.14 jd 22'
alias ssh_tx='expect $perl_p/ssh_.exp root 119.23.8.57 BGILYLWly61 22'
alias ssh_pi='expect $perl_p/ssh_.exp pi algoers.com pi 10240'
#alias ssh_pi='expect $perl_p/ssh_.exp pi lewelab.com pi 10240'
alias mo_u='module unload '
alias mo_l='module load '
alias mo_a='module available '
alias ARChitect_v='ARChitect2 -cl -v '
alias rm~='rm -f *~ '
alias rm='rm -f '
alias p_o='p4 open '
alias p_od='p4 opened '
alias find_by_name='find `pwd` -name '
alias findf_by_name='find `pwd` -type f -name '
alias lshp='find `pwd` -type f -name "*" | xargs ls -th | head '
alias c='clear '
alias objd='objdump '
alias mdb_run_elf='mdb -run -av2hs -cl '
alias mdb_dbg_elf='mdb -cl -av2hs '
alias perl_create_makefile=' perl $perl_p/auto_create_makefile_OK.PL > Makefile && echo "create Makefile and [make all] " && make all '
alias perl_show_files_content='perl $perl_p/show_file_content.PL '
alias perl_mqx_config='yes|cp -u $perl_p/mqx_*config*.PL ./mqx_config && perl mqx_config '
alias cp_to_bak='sh $perl_p/cp_to_bak.sh '
alias ia_2013='/slowfs/us01dwt2p448/flexera/InstallAnywhere_2013/InstallAnywhere'
alias ia_2014='/slowfs/us01dwt2p448/flexera/InstallAnywhere_2014-SP1/InstallAnywhere'
alias ia_build='/slowfs/us01dwt2p448/flexera/InstallAnywhere_2014-SP1/build'
alias RE='cat $perl_p/README '
alias s_bashrc='source ~/.bashrc'
alias s_b='source ~/.bashrc'
alias cygpath='$perl_p/cygpath_aw.sh '
alias archi_env_set='sh $perl_p/architect_set_env.sh '
alias perl_diff_foler='perl $perl_p/diff_ia_folder_R1_R0.PL '
alias lsh=' perl $perl_p/lsh.PL ' 
alias readlink_file=' perl $perl_p/readlink_file.PL ' 
alias see_path='sh $perl_p/see_path.sh '
alias scp_it='perl $perl_p/scp_it.PL '
alias full_path='perl $perl_p/full_path.PL '
alias full_='perl $perl_p/full_path.PL '
alias ff='perl $perl_p/ff.PL ' 
alias tol='perl $perl_p/tol.PL ' 

alias fr='perl $perl_p/fr.PL'
alias tor='perl $perl_p/tor.PL'

alias peval='perl $perl_p/peval.PL ' 
alias latest_info='(basename `full_ $latest_mwdt` && basename `dirname \`full_ $latest_mide\`` ) | tee $tmp/perl_p/latest_info.log'
alias s_l='source ~/txt.txt && export PATH=/SCRATCH/ARC/ARC_/MetaWare/ide:$PATH'
alias setenv_latest_mwdt_mide='latest_info && perl $perl_p/setenv_latest_mwdt_mide.PL && s_l '
alias setenv_daily_mide='perl $perl_p/setenv_daily_mide.PL '
alias cc_test='perl $perl_p/cc_test_linux.PL '
alias ecd='perl $perl_p/ecd.PL '
alias qftest='${qftest_root}/qftest -license ~linqi/license/qft_lic.dat '
alias qft_code_gen=' cd $perl_p && perl $perl_p/qft_code_gen.PL '
alias gen_qft_code=' perl $perl_p/gen_qft_code.PL '
alias g='gedit '
alias h='history '
alias chmod_r='chmod -R 0775 `pwd` '
alias QSUB='qsub -P bnormal -cwd -V -l arch=glinux,os_bit=64,cputype=emt64,os_distribution=redhat'
#get an xterm
alias xlight='qsh -P ilight -display $DISPLAY_USE -l arch=glinux,os_bit=64 -- -bg Azure1'
alias xheavy='qsh -P iheavy -l os_version=WS6.0'
alias xheavy32='qsh -P iheavy -l os_version=WS6.0,os_bit=32'
#bsub an interactive job
alias qheavy='qrsh -P iheavy -l os_version=WS4.0 -now no -b y'
alias qft_env_='cat $qft_mide/script/*.sh |grep env_ '
alias cbin='perl $perl_p/cbin.PL ' 
alias mc='cd $rmo && sh startup_mongo.sh ' 
alias map_='perl $perl_p/map_.PL ' 
alias psf='ps -u $USER -f ' 
alias nll='nl -w 4 -b a -n rz ' 
alias t='cd $t'
alias et='cd $et'
alias gs='cd $gs'
alias gss='cd $gss'
alias gss_='cd $gss_'
alias db='cd $db'

alias dg='newgrp - docker'
alias m_a2='perl $perl_p/m_a2.PL '

### new added ###
alias git_push='perl $perl_p/git_push.PL ';
alias git_push_no_pw='perl $perl_p/git_push_no_pw.PL ';
alias gmain='gss_ && cp_to_bak main.cpp && rm main.cpp && git checkout --theirs main.cpp && git pull'
alias mail_bgi='perl $perl_p/mail_bgi.PL '
alias mount_sf_et='perl $perl_p/mount_sf_et.PL'
alias bwa='cd $bwa'
alias L='perl $perl_p/L.PL'
alias psh='ps au --sort=lstart'
alias seq_2_num='perl $perl_p/seq_2_num.PL'
#alias tfr='docker exec ub_ssh_v4_0 bash /usr/bin/tfr.sh '
