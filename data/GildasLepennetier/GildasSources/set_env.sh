#!/bin/bash

################################################################################
#
#	Install tools for a functional Linux environment
#
#	apt: Advanced Package Tool
#
#	Copyright 2015 LEPENNETIER Gildas
#	Author: Gildas Lepennetier - gildas.lepennetier@hotmail.fr
#
#																2 July 2015
################################################################################

################################################################################
#
#			options
#
#set -e					# to stop this script if error
workDir=$(pwd)				# current working directory
yes="--assume-yes"			# to say yes during install
WGET="wget -N"			# option for the wget: Turn on time-stamping, download if new version, and quiet
processorType=$(arch)		# processor type: i386, amd64...
if [[ -z "$processorType" ]];then echo "architecture not found with arch command: please enter: i386, i686 or x86_64..."; read processorType; fi
archi=$(getconf LONG_BIT)	# computer's architechture: return 32 or 64
OS=$(uname -s)
L=""						# list of programs to install
################################################################################

# OTHER packages managers
#sudo apt-get install rpm	#package manager
#sudo apt-get install yum	#yum updater

echo -e "\nInstall system update? (yes)"
read q
if [ "$q" == "yes" ];then echo -e "\n\nStarting: $0 ... system update and set up\n"

	if [ 0 ];then echo -e "\n added: system (compiler and compatibility)"
		#L="$L htop"	#display system daemons
		L="$L g++"	#C compiler
		L="$L wine"	#wine windows compatibility
	fi

	# pythttp://tandem.bu.edu/trf/trf407b.linux.download.htmlhon packages
	if [ 0 ];then echo -e "\n added: python extra tools"
		L="$L python-dev python-setuptools"	#python setup tools: sudo easy_install
		L="$L python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose"	# scipy and numpy
		L="$L python-biopython python-biopython-doc python-biopython-sql"	# biopython
		L="$L idle idle3" # IDLE for python3
		L="$L python-pip"	# pip for cutadapt
		L="$L python3-pip"	# pip for python3
	fi

	# SSH
	if [ 0 ];then echo -e "\n added: ssh and sshd"
		L="$L openssh-server"
	fi

	#internet tools
	if [ 0 ];then echo -e "\n added: internet tools"
		L="$L curl"	#internet function: download pages
		L="$L whois"	#link ip and domain
		L="$L nmap"	#Network Mapper
		#L="$L apache2-bin" # Apache system ???
	fi

	#bioinfo
	if [ 0 ];then echo -e "\n added: (bio)informatic tools"
		L="$L libncurses-dev"			# required by tuxedo ?
		L="$L tophat bowtie cufflinks"	# tuxedo programs
		L="$L bedtools"		http://tandem.bu.edu/trf/trf407b.linux.download.html			# bam to fastq
		# blast
		L="$L ncbi-blast+"
	fi
	
	#utilities
	if [ 0 ];then echo -e "\n added: #utilities"
		L="$L parallel"			# to perform parallel operations

	fi
	
	
	# Thunderbird mail agent
	if [ 0 ];then echo -e "\n added: Thunderbird"
		L="$L thunderbird"
	fi

	# Filezilla FTP agent
	if [ 0 ];then echo -e "\n added: Filezilla"
		L="$L filezilla"
	fi

	#
	echo "$L"
	echo "install those packages? (yes)"
	read q
	if [ "$q" == "yes" ]; then sudo apt-get $yes install $L ; else echo "Cancelled"; fi

	# fix dependencies problems
	sudo apt-get install -f
	#sudo apt-get $yes build-dep "package" #specifically build dependencies for the package

fi

# RepeatMasker
echo -e "\nInstall RepeatMasker? (yes)"
read q
if [ "$q" == "yes" ];then cd $HOME; echo -e "\nRepeatMasker, libraries and search programs"
	
	#echo "todo: repeatmasker in a better place than home, same for hmmer and so one"
	
	if [ ! -d $HOME/trf ];then echo "init -- get tandem repeat finder - trf"
		mkdir -p "$HOME/trf"
		echo "1) Go to : http://tandem.bu.edu/trf/trf407b.linux.download.html"
		echo "2) Download the program by clicking on the download button (name: trf407b.linux)"
		echo "3) Put it in the directory called 'trf' in your home (it was created)"
		echo "4) When done, press here enter."
		echo "trf address should be: $HOME/trf/trf407b.linux the program called trf"
		echo "the programm will be renamed trf, since it is required by RepeatMasker"
		read X
		mv -v $HOME/trf/trf407b.linux $HOME/trf/trf
		sudo chmod +x $HOME/trf/trf
	fi
	
	if [ ! -d $HOME/RepeatMasker ];then echo "get RepeatMasker"
		$WGET "http://www.repeatmasker.org/RepeatMasker-open-4-0-5.tar.gz"
		tar -xf RepeatMasker-open-4-0-5.tar.gz
		rm RepeatMasker-open-4-0-5.tar.gz
	fi

	if [ 0 ];then echo "install Hmmer"
		cd $HOME
		VERSION_Hmmer=hmmer-3.1b2-linux-intel-x86_64
		echo -e "\n!! I need the name of the most up-to-date program !! \n\nExemple: '$VERSION_Hmmer' \nplease enter program name: (ENTER=default)"
		read VERSION_Hmmer
		if [ -z "$VERSION_Hmmer" ];then VERSION_Hmmer=hmmer-3.1b2-linux-intel-x86_64; fi
		
		$WGET "http://eddylab.org/software/hmmer3/3.1b2/$VERSION_Hmmer.tar.gz"
		tar -xvf $VERSION_Hmmer.tar.gz
		cd $VERSION_Hmmer
		
		./configure
		make #build
		make check #automated tests
		echo "installation of hammer requires root"
		sudo make install #automated install
		cd $HOME
		rm $VERSION_Hmmer.tar.gz
		rm -r $VERSION_Hmmer
		
		echo -e "\n\nHMMER in /usr/local/bin (if everything was fine during install)"
	fi
	
	if [ 0 ];then echo "search program RMVlast / Blast"
		
		
		cd $HOME
		echo "downloading RMBlast Binaries"
		VERSION_RMBlast=ncbi-rmblastn-2.2.28-x64-linux.tar.gz
		echo -e "\n!! I need the name of the most up-to-date program !! \n\nExemple: '$VERSION_RMBlast' found at ftp://ftp.ncbi.nlm.nih.gov/blast/executables/rmblast/LATEST\nplease enter program name: (ENTER=default)"
		read VERSION_RMBlast
		if [ -z "$VERSION_RMBlast" ];then VERSION_RMBlast=ncbi-rmblastn-2.2.28-x64-linux.tar.gz; fi
		$WGET "ftp://ftp.ncbi.nlm.nih.gov/blast/executables/rmblast/LATEST/$VERSION_RMBlast"
		tar zxvf $VERSION_RMBlast
		rm $VERSION_RMBlast
		mkdir -p $HOME/bin/
		cd ncbi-rmblastn*/bin
		sudo chmod +x *
		cp * $HOME/bin//usr/local/bin
		
		
		cd $HOME
		echo "downloading BLAST+ Binaries"
		VERSION_blast=ncbi-blast-2.3.0+-x64-linux.tar.gz
		echo -e "\n!! I need the name of the most up-to-date program !! \n\nExemple: '$VERSION_blast' found at ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST\nplease enter program name: (ENTER=default)"
		read VERSION_blast
		if [ -z "$VERSION_blast" ];then VERSION_blast=ncbi-blast-2.3.0+-x64-linux.tar.gz; fi
		$WGET ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/$VERSION_blast
		tar zxvf $VERSION_blast
		rm $VERSION_blast
		cd $HOME/ncbi-blast*/bin/
		sudo chmod +x *
		cp -v $HOME/ncbi-rmblastn-*/bin/rmblastn .
		
		echo "move everything in usr/local/bin"
		mkdir -p $HOME/bin/
		cp $HOME/${VERSION_blast%+*}+/bin/* $HOME/bin/
		
		echo -e "\n\nALL BINARIES ARE IN $HOME/bin --don't forget to put it in the path if you wana use blast or the other program later!\n\n"
		
		echo "you can also remove the ncbi-blast-*  and  ncbi-rmblastn-* folders in your home"
		read -p "execute rm -vr ncbi-blast-* ncbi-rmblastn-* ? (yes)" yesno
		if [ "$yesno" == "yes" ]; then rm -vr $HOME/ncbi-blast-* $HOME/ncbi-rmblastn-*; fi
		
	fi
	
	if [ 0 ];then echo "downloading Libraries"
		cd $HOME/RepeatMasker
		#repbase require login
		echo -e "you should register to access the RepBase libraries. it is not allowed to use my login.\n\tRegister here: http://www.girinst.org/accountservices/register.php";
		read -p "your RepBase login (mine is gildas): " LOGIN
		read -p "your RepBase password (mine is edr6qv): " PASS
		VERSION_lib=20.12
		read -p "please enter version for libraries exemple: '$VERSION_lib' ( @ http://www.girinst.org ):" VERSION_lib
		if [[ -z "$VERSION_lib" ]];then VERSION_lib=20.12; fi
		$WGET --user=$LOGIN --password=$PASS http://www.girinst.org/server/RepBase/protected/RepBase$VERSION_lib.embl.tar.gz
		
		tar -xvf RepBase$VERSION_lib.embl.tar.gz
		rm RepBase$VERSION_lib.embl.tar.gz
		#$WGET --user=$LOGIN --password=$PASS http://www.girinst.org/server/RepBase/protected/repeatmaskerlibraries/repeatmaskerlibraries-20140131.tar.gz
		#tar -xvf repeatmaskerlibraries-20140131.tar.gz
		#$WGET --user=$LOGIN --password=$PASS http://www.girinst.org/server/RepBase/protected/REPET/RepBase$VERSION"_REPET.embl.tar.gz"
		#tar -xvf RepBase19.06_REPET.embl.tar.gz
		$WGET --user=$LOGIN --password=$PASS http://www.girinst.org/server/RepBase/openaccess/dfamrepref.embl.tgz
		tar -xvf dfamrepref.embl.tgz
		rm dfamrepref.embl.tgz
		### duplicate with embl
		### $WGET "--user=$LOGIN --password=$PASS http://www.girinst.org/server/RepBase/protected/RepBase19.12.fasta.tar.gz"
		### tar -xvf RepBase19.12.fasta.tar.gz
	fi
	
	echo "configure RepeatMasker"
	
	#sudo find / -name nhmmer
	echo "perl: /usr/bin/perl"
	echo "RepeatMasker: $HOME/RepeatMasker"
	
	echo "trf: $HOME/trf"
	echo "hammer: /usr/local/bin"
	echo "rmblast: $HOME/bin"
	
	read -p "remember those addresses, the configuration will start after you press enter" X
	cd $HOME/RepeatMasker/
	perl configure
	#Further documentation on the program may be found here: /home/gildas/RepeatMasker/repeatmasker.help
fi

# R
echo -e "\nInstall R? (yes)"
read q
if [ "$q" == "yes" ];then cd $HOME
	
	#dependencies required
	sudo apt-get install libjpeg62 $Y 
	
	#check infos
	#https://cran.uni-muenster.de/bin/linux/ubuntu/
	
	echo "add repository for updates using the following depository: https://cran.uni-muenster.de/"
	sudo add-apt-repository "deb https://cran.uni-muenster.de/bin/linux/ubuntu trusty/"
	
	#The Ubuntu archives on CRAN are signed with the key of “Michael Rutter marutter@gmail.com” with key ID E084DAB9
	sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
	
	# R
	sudo apt-get install r-base r-base-dev r-base-html r-doc-html $Y  #libjpeg62 required
	
	# maintenance later:
	#sudo apt-get update
	#sudo apt-get upgrade

	
	#Users who need to compile R packages from source [e.g. package maintainers, or anyone installing packages with install.packages()] should also install the r-base-dev package:
	#sudo apt-get install r-base-dev

fi

echo -e "\nInstall RStudio? (yes)"
read q
if [ "$q" == "yes" ];then cd $HOME
	# RStudio
	#VERSION=0.99.473 # August 2015
	VERSION="$(curl -s "https://www.rstudio.com/products/rstudio/release-notes/" | grep "<h2>RStudio v" | awk '{print $2}' | sed -e 's/v//')"
	#wget -q some-url -O - | grep something
	
	if [ -z "$VERSION" ];then echo "Could not find automaticaly RStudio version, please check https://www.rstudio.com/products/rstudio and enter 0.99.473"; read VERSION; fi
	
	if [ "$processorType" == "i386" ];then V="$VERSION-i386"; fi
	if [ "$processorType" == "i686" ];then V="$VERSION-i386"; fi #http://download1.rstudio.org/rstudio-0.99.441-i386.deb
	if [ "$processorType" == "x86_64" ];then V="$VERSION-amd64"; fi #http://download1.rstudio.org/rstudio-0.99.441-amd64.deb
	$WGET "http://download1.rstudio.org/rstudio-$V.deb"
	sudo dpkg -i rstudio-$V.deb
	rm rstudio-$V.deb
	
	if [ ];then 
			
		#java compiler ? r-java
		#sudo apt-get install gcj-aarch64-linux-gnu r-cran-rjava
		
		#exporting variable
		#R CMD javareconf -e
	
		echo "run sudo R"
		
		#for Bioconductor
# 		source("http://bioconductor.org/biocLite.R")
# 		biocLite("BiocUpgrade")
# 			#biocValid()
# 			#biocLite()
# 		biocLite("cummeRbund") #Sequencing
# 			#biocLite("ArrayExpressHTS")
# 		biocLite("QuasR")
# 			#biocLite("rjava")
# 		biocLite("limma") #for Venn Diagram
# 		
# 		#PACKAGES default lib=/home/gildas/R/x86_64-pc-linux-gnu-library/3.1
# 		install.packages("ggplot2",dependencies=T,repos="http://mirrors.softliste.de/cran/")			#plot
# 		install.packages("zoo",dependencies=T,repos="http://mirrors.softliste.de/cran/")				#rolling function
# 		install.packages("plyr",dependencies=T,repos="http://mirrors.softliste.de/cran/")	        	#arrange dataframe
# 		install.packages("gtools",dependencies=T,repos="http://mirrors.softliste.de/cran/")	        	#cut a vector depending on quantiles
# 		install.packages("ppcor",dependencies=T,repos="http://mirrors.softliste.de/cran/")	        	#partial corelation
# 		install.packages("FactoMineR",dependencies=T,repos="http://mirrors.softliste.de/cran/")			#PCA
# 		install.packages("ade4",dependencies=T,repos="http://mirrors.softliste.de/cran/")	        	#PCA
# 		install.packages("rgl",dependencies=T,repos="http://mirrors.softliste.de/cran/")				#repository=Berlin #plot3d, X11 and header GL/gl.h required
# 		install.packages("argparse",dependencies=T,repos="http://mirrors.softliste.de/cran/")			#argument parser
# 		install.packages("vegan",dependencies=T,repos="http://mirrors.softliste.de/cran/")				#with Karsten
# 		install.packages("rococo",dependencies=T,repos="http://mirrors.softliste.de/cran/")				#Robust Rank Correlation Coefficient test
# 		install.packages("boot",dependencies=T,repos="http://mirrors.softliste.de/cran/")				# bootstrap
# 		install.packages("raster",dependencies=T,repos="http://mirrors.softliste.de/cran/")				# modal value, ...
# 		install.packages("VennDiagram",dependencies=T,repos="http://mirrors.softliste.de/cran/")				# Venn Diagram
# 		package for map and GPS handling?
	fi

fi

# imageMagick - tiff support
# imageMagick - cmd line
echo -e "\nInstall imageMagick? (yes)"
read q
if [ "$q" == "yes" ];then cd $HOME
	
	if [ 0 ];then echo -e "\n tiff support - need to be installed first"
		
		rm -fr tiff-4.0.4 #because we want clean
		
		$WGET "ftp://ftp.remotesensing.org/libtiff/tiff-4.0.4.tar.gz"
		tar xzf "tiff-4.0.4.tar.gz"
		rm -v "tiff-4.0.4.tar.gz"
		cd "tiff-4.0.4"
		
		#echo "patching the stuff...this was for version 4.0.3"
		#$WGET "http://www.linuxfromscratch.org/patches/blfs/svn/tiff-4.0.3-fixes-1.patch"
		#patch -Np1 -i tiff-4.0.3-fixes-1.patch
		#rm tiff-4.0.3-fixes-1.patch
		
		./configure
		sudo make install
		make check
		make clean
		
		#Libraries have been installed in: /usr/local/lib
# 		If you ever happen to want to link against installed libraries
# 		in a given directory, LIBDIR, you must either use libtool, and
# 		specify the full pathname of the library, or use the '-LLIBDIR'
# 		flag during linking and do at least one of the following:
# 		- add LIBDIR to the 'LD_LIBRARY_PATH' environment variable
# 			during execution
# 		- add LIBDIR to the 'LD_RUN_PATH' environment variable
# 			during linking
# 		- use the '-Wl,-rpath -Wl,LIBDIR' linker flag
# 		- have your system administrator add LIBDIR to '/etc/ld.so.conf'
# 
# 		See any operating system documentation about shared libraries for
# 		more information, such as the ld(1) and ld.so(8) manual pages.
		
		cd $HOME
		sudo rm -rv tiff-4.0.4
	fi
	
	sudo apt-get install imagemagick $Y 
	sudo ldconfig /usr/local/lib #to fix shared library problems
	
	#check some variables
	#convert -list configure
	#convert -list format
	
	if [ ];then echo "manualy, compile from sources (hope it works, if the previous option do not)"
		cd $HOME
		VERSION_MAGICK="6.9.1-3"
		$WGET "ftp://ftp.imagemagick.org/pub/ImageMagick/ImageMagick-$VERSION_MAGICK.tar.gz"
		
		#ftp://ftp.imagemagick.org/pub/ImageMagick/ImageMagick.tar.gz
		
		tar xzf "ImageMagick-$VERSION_MAGICK.tar.gz"
		rm -v "ImageMagick-$VERSION_MAGICK.tar.gz"
		cd "ImageMagick-$VERSION_MAGICK"
		./configure
		make check
		sudo make install
		sudo ldconfig /usr/local/lib #to fix shared library problems
		make clean
		
		#http://www.imagemagick.org/script/command-line-tools.php
		#http://www.imagemagick.org/Usage/
	fi
	
fi

# Zotero
echo -e "\nInstall zotero? (yes)"
read q
if [ "$q" == "yes" ];then cd $HOME
	
# 	echo "1) go to https://www.zotero.org/download/"

# BETTER: make a laucher, don't forget that run-zotero.sh want you to be in the surrent directory to start properly

# 	echo "2) save the file (having the .tar.bz2) in $HOME and press ENTER when finished"
# 	read q
# 	echo "3) extracting the archive"
# 	tar -xjf Zotero*.tar.bz2
# 	rm -vr Zotero*.tar.bz2
#make a 
	
	read -p "Do you want to add zotero to PATH? (yes)" q
	if [ "$q" == "yes" ];then
		cd $HOME/Zotero*
		chmod +x run-zotero.sh
		echo -e "\n#Zotero - added $(date)\nPATH=\$PATH:$(pwd)" >> "$HOME/.bashrc" #pwd is supposed to be something like /home/gildas/Zotero_linux-i686
		PATH=$PATH:$(pwd)
		echo -e "\nTo run Zotero: run-zotero.sh"
	fi
fi


# Dropbox
if [ ];then cd $HOME; echo -e "\n install: Dropbox"
	sudo apt-get install nemo python-gpgme libnemo-extension1 nemo-dropbox $Y 
	echo "need to quit the daemon, and restart/start"
	echo "nemo --quit "
	echo "dropbox start -i"
	#gilou_on_net@hotmail.com
	#sy...
fi

################################################################################

## FIREFOX sync: gildas.lepennetier@hotmail.fr

# cmd to know - MEMO
if [ ];then cd $HOME; echo -e "\n install: commands to know / master"
	### commands to know
	sudo apt-get install -f	# fix brocken packages
	sudo apt-get update		# update
	sudo apt-get upgrade $Y	# upgrade
	sudo apt-get autoremove	# remove

	hostname NewName		# change computer name
	lscpu				# cpu infos

	echo uname #unix name
	echo arch #achitecture
	echo rsync #synchronization
	echo fdisk #checkdisk

	echo which #where is the command located
	echo du #disl use
	echo df #disk free
	echo dd #copy data
	echo find #do not forget the -name option
	echo locate #using index to search

	echo !! #!! is replaced by the previous command / potential error of execution: echo $(!!)
	echo useradd and adduser

	#gestion of the volume sound
	echo alsamixer
	
	echo ping 128.176.213.35 -c 5 #ping my office
fi
# tips, techniques and so on
if [ ];then cd $HOME; echo -e "\n install: little notes"
	exit
	############################################################################
	######## mount a disk
	# UUID="10E82EB2E82E964E"
	# type="ntfs"
	# dir="/media/Daten"
	# sudo mkdir -p $dir
	# sudo mount -t $type -U $UUID $dir
	######## make it auto
	#sudo nano /etc/fstab # mount automatically Daten ( DIRECTORY must exist)
	#UUID=10E82EB2E82E964E /media/Daten ntfs defaults 0 2
	############################################################################
	######## which user are you ?
	if [[ ${EUID} == 0 ]] ; then echo "you are root";else echo "you are not root, but user: $EUID"; fi
	[ ${EUID} == 0 ] && echo "you are root" || echo "you are not root, but user: $EUID"
	############################################################################
	######## colors in terminal
	#https://wiki.archlinux.org/index.php/Color_Bash_Prompt#Basic_prompts
	# in nano /etc/bash.bashrc
	# echo $PS1 #colors for terminal
	#PS1="\[\033[0;31m\]$(returncode)\[\033[0;37m\]\[\033[0;35m\]${debian_chroot:+($debian_chroot)}\[\033[0;35m\]\u@\h\[\033[0;37m\]:\[\033[0;36m\]\w >\[\033[0;00m\]"
	# put everything in blue
	#PS1="\[\033[34m\][\$(date +%H%M)][\u@\h:\w]$ "
	############################################################################
	############################################################################
	# PBS (portable batch system) :http://dcwww.camd.dtu.dk/pbs.html
	#How to:
	#command for submission is
	qsub subfile
	############################################################################
	#subfile:
	#!/bin/sh
	### Job name
	#PBS -N test
	### Declare job non-rerunable
	#PBS -r n
	### Output files
	#PBS -e test.err
	#PBS -o test.log
	### Mail to user
	#PBS -m ae
	### Queue name (small, medium, long, verylong)
	#PBS -q long
	### Number of nodes (node property ev67 wanted)
	#PBS -l nodes=8:ev67
	# This job's working directory
	echo Working directory is $PBS_O_WORKDIR
	cd $PBS_O_WORKDIR
	echo Running on host `hostname`
	echo Time is `date`
	echo Directory is `pwd`
	echo This jobs runs on the following processors:
	echo `cat $PBS_NODEFILE`
	# Define number of processors
	NPROCS=`wc -l < $PBS_NODEFILE`
	echo This job has allocated $NPROCS nodes
	# Run the parallel MPI executable "a.out"
	mpirun -v -machinefile $PBS_NODEFILE -np $NPROCS a.out
	############################################################################
fi

#
# non-necessary / memo
#
# libreoffice - manual installation
if [ ];then cd $HOME; echo -e "\n install: libreoffice"
	#install kde integration for libreoffice
	sudo apt-get install libreoffice-kde
	#download libreoffice - last version
	read -p "Did you update the version in the script?"
	$WGET "http://donate.libreoffice.org/fr/dl/deb-x86_64/4.3.7/fr/LibreOffice_4.3.7_Linux_x86-64_deb.tar.gz"
	tar -xvzf LibreOffice_4.3.7_Linux_x86-64_deb.tar.gz
	cd LibreOffice_4.3.7.2_Linux_x86-64_deb/DEBS
	sudo dpkg -i *.deb
	cd ../..
	rm -r LibreOffice_4.3.7.2_Linux_x86-64_deb
	rm LibreOffice_4.3.7_Linux_x86-64_deb.tar.gz
fi

#sudo apt-get install gnumeric	#excel 



# PERL
if [ ];then cd $HOME; echo -e "\n install: perl modules"
	sudo apt-get install perl-doc $Y 
	#search modules -> http://search.cpan.org/
	# perl modules
	sudo cpan install Base
	sudo cpan HTTP
	echo "TODO: install CPAN"
	echo "TODO: reload cpan"
fi


# git
if [ ];then cd $HOME; echo -e "\n install: Git - supposed to be there already"
	sudo apt-get install git $Y
	
	#exemple: clone the pirate bay
	#git clone https://github.com/isohuntto/openbay.git
	#git clone https://github.com/GildasLepennetier/GildasSources.git
	#git clone https://github.com/GildasLepennetier/smartPySpider.git
fi

if [ ];then echo "other"
	sudo easy_install weblogo			# weblogo
	sudo easy_install --upgrade weblogo
	sudo pip install cutadapt			# cutadapt
	
	# blat for x86_64
	$WGET "http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/blat/blat"
	sudo chmod +x blat
	sudo mv blat /usr/bin/blat
	
	
	sudo apt-get install mysql-server
	
	#sudo apt-get install pinta $Y #paint-like
	#sudo apt-get install freemind $Y 
	
	#midnight commander
	#sudo apt-get install mc $Y
	
	#L="$L dmidecode"	#view BIOS Information
	#L="$L baobab"	#disc usage
	
	
	#sound problem : cannot open mixer: No such file or directory
	#lspci -nnk | grep -iEA3 audio
	#mplayer -i audio.mp3
	#lsmod | grep snd
fi