#!/bin/sh

username=$(whoami)
export HOME="/lustre/gk36/$username"
export PATH=$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 自分のプロジェクトに合わせて環境変数をセットする
module load gnu/gcc_7.5.0
