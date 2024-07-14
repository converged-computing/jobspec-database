#!/bin/sh

yyyymm01=${1}

monthlen=`date -ud "${yyyymm01} +1month -1days" +%d`
((monthlen--))
MM=`date -ud "${yyyymm01}" +%m`

for filetype in emis_mole_all #emis_mole_all_shipping
do
for i in $( eval echo {7..${monthlen}} )
do
    today=`date -ud "${yyyymm01} +${i}days" +%Y%b%a`
    todayi=`date -ud "${yyyymm01} +${i}days" +%Y%m%d`
    echo $todayi $today
    for frep in `ls rep${filetype}.2019${MM}0[1234567].nc`
    do
        repdayi=${frep: -11:-3}
        repday=`date -ud $repdayi +%Y%b%a`
        if [ $repday = $today ]; then
            echo A match!
            echo $today MATCHES $repday
            echo $frep
            ln -s $frep rep${filetype}.${todayi}.nc
            echo ''
            continue
        fi
    done
done
done
