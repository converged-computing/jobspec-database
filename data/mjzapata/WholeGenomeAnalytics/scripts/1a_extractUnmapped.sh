#!/bin/bash
. wgaconfig.conf
mkdir ${scripts1aDir}
mkdir ${unmappedDir}
# hack near the bottom, selecting only the first 0:9 characters of the file basename
# difference between bamResults and Sorted??
i=0
while read line
do
bamFiles[ $i ]="$line"
(( i++ ))
done < <(find ${bamDir}*.bam -type f -exec basename {} \;)
arrayLength=${#bamFiles[@]}
echo $arrayLength "files"

#for (( k=0; k < $arrayLength; k++))
#do
#echo $k
#echo ${bamFiles[k]:0:9}
#done


for (( i=0; i < $arrayLength; i++))
do
cat > ${scripts1aDir}pbsextract_$i << EOF
#!/bin/bash
#PBS -q viper
#PBS -N mjzExtractUnmappedReads_${i}
#PBS -l walltime=20:01:00
#PBS -l nodes=1
${samtoolsDir}samtools view -b -f 8 ${bamDir}${bamFiles[i]} | java -jar ${picardDir}SamToFastq.jar I=/dev/stdin F=${unmappedDir}${bamFiles[i]}_1.fastq F2=${unmappedDir}${bamFiles[i]}_2.fastq
EOF
done



# output unmapped reads to paired