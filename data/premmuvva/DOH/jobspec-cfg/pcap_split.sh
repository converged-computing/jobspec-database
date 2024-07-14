max_threads=10
current_threads=0

find "$1" -type f -print0 | while IFS= read -r -d '' file_path;
do
    echo File : $file_path
    if [ -f "$file_path" ] && [[ "$file_path" == *.pcap ]]; then
        for stream in `~/tshark/usr/local/bin/tshark -r $file_path -T fields -e tcp.stream | sort -n | uniq`
        do
            echo "processing stream number: $stream"
            # outputfile=$(echo "$1" | awk -F'/' '{print $2}' | awk -F'.' '{print $1}')
            outputfile=$(basename "$file_path" | cut -f 1 -d '.')
            echo "$outputfile $file_path"
            {
            ~/tshark/usr/local/bin/tshark -r $file_path -w $2/$outputfile-$stream.pcap -Y "tcp.stream==$stream && tls.app_data" 
            ((current_threads--))
            } &
            ((current_threads++))
            if [ "$current_threads" -ge "$max_threads" ]; then
                wait
                current_threads=0
            fi
        done
    fi
done