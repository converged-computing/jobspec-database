#!/bin/bash

python sklearn-kmeans -f otto_group_61878_93.csv -k 5  -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 10 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 15 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 20 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 25 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 30 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 35 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 40 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 45 -t 500 -p $1
python sklearn-kmeans -f otto_group_61878_93.csv -k 50 -t 500 -p $1
