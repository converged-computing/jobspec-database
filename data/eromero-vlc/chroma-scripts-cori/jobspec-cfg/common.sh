
# num_args ...
# Return the number of arguments

num_args() {
	echo $#
}

# take_first ...
# Return the first argument
take_first() {
	echo ${1}
}

# mom_word momx0 momy0 momz0 [momx1 momy1 momz1]
# Return a single word representing a momentum (transfer)

mom_word() {
	[ $# == 3 ] && echo ${1}_${2}_${3}
	[ $# == 6 ] && echo ${1}_${2}_${3}_${4}_${5}_${6}
}

# mom_fly momx0 momy0 momz0 [momx1 momy1 momz1]
# Return a canonical direction of mom0 - mom1

mom_fly() {
	if [ $# == 3 ]; then
		if [ $1 -gt 0 -o $2 -gt 0 -o $3 -ge 0 ]; then
			echo $1 $2 $3
		else
			echo $(( -$1 )) $(( -$2 )) $(( -$3 ))
		fi
	else
		if [ $1 -gt $4 ] || [ $1 -eq $4 -a $2 -gt $5 ] || [ $1 -eq $4 -a $2 -eq $5 -a $3 -ge $6 ]; then
			echo $(( $1-$4 )) $(( $2-$5 )) $(( $3-$6 ))
		else
			echo $(( $4-$1 )) $(( $5-$2 )) $(( $6-$3 ))
		fi
	fi
}

# mom_split momx0 momy0 momz0 [momx1 momy1 momz1]
# Return all momenta needed for a canonical momentum

mom_split() {
	if [ $# == 3 ]; then
		echo $1 $2 $3
		echo $(( -$1 )) $(( -$2 )) $(( -$3 ))
	else
		echo $1 $2 $3
		echo $4 $5 $6
		echo $(( -$1 )) $(( -$2 )) $(( -$3 ))
		echo $(( -$4 )) $(( -$5 )) $(( -$6 ))
	fi
}

# shuffle_t_source cfg [t_size t_source]
shuffle_t_source() {
	local cfg t_size t_source t_shift
	cfg="$1"
	t_size="${2:-0}"
	t_source="${3:-0}"
	t_shift="$( perl -e " 
  srand($cfg);

  # Call a few to clear out junk                                                                                                          
  foreach \$i (1 .. 20)
  {
    rand(1.0);
  }
  \$t_shift = int(rand($t_size));
  print \"\$t_shift\\n\"
")"
	if [ $t_size == 0 ]; then
		echo $t_shift
	else
		echo "$(( (t_source + t_shift) % t_size ))"
	fi
}

# k_split n args...
# Return args... broken in different lines with up to <n> elements in each line
k_split() {
	local n i f
	n="$1"
	shift
	i="0"
	for f in "$@" "__last_file__"; do
		if [ $f != "__last_file__" ]; then
			echo -n "$f "
			i="$(( i+1 ))"
			if [ $i == $n ]; then
				i="0"
				echo
			fi
		else
			[ $i != 0 ] && echo
		fi
	done
}

# k_split n args...
# Return args... broken in different up to <n> lines
k_split_lines() {
	local n i f num_args line
	n="$1"
	shift
	num_args="$#"
	i=0
	line=0
	for f in "$@" "__last_file__"; do
		if [ $f != "__last_file__" ]; then
			echo -n "$f "
			i="$(( i+1 ))"
			if [ $i == $(( num_args/n + (line<num_args%n ? 1 : 0) )) ]; then
				i="0"
				line="$(( line+1 ))"
				echo
			fi
		else
			[ $i != 0 ] && echo
		fi
	done
}
