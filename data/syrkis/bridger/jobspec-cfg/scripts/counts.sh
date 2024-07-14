#! /usr/bin/zsh

for F in $(ls /home/common/datasets/amazon_review_data_2018/reviews/); do
	if [ ! -f "/home/$(whoami)/bridger/data/counts/$(echo $F | cut -d '.' -f 1).out" ]; then
		echo "begin $F"
		date
		zcat /home/common/datasets/amazon_review_data_2018/reviews/$F | jq '.["reviewText"]' | tr 'a-z' 'A-Z' | tr -sC 'A-Z' ' ' | tr -s ' ' '\n' | sort | uniq -c | sort -n > /home/$(whoami)/bridger/data/counts/$(echo $F | cut -d '.' -f 1).out
		echo "JUST DID $F"
		date
	else
		echo "ALREADY DONE $F"	
		cat /home/$(whoami)/bridger/data/counts/$(echo $F | cut -d '.' -f 1).out | sed 's/^ *//' | cut -d ' ' -f 1 | paste -sd+ | bc >> /home/$(whoami)/bridger/data/counts/$(echo $F | cut -d '.' -f 1).out
		date
	fi
done

echo "GOD BLESS THE MACHINE"
