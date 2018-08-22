#!/bin/bash -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/willy/.mujoco/mjpro150/bin

env=$1
network_size=$2
scale_interval=$3
scale_threshold=$4
network_ratio=$5
r=$6

dir=$env-network_$network_size-network_ratio_$network_ratio-reward_scaling-$r

python main.py --env-name "$env"  \
               --num-stack 1  \
               --hidden_size $network_size \
               --num-frames 100000000  \
               --log-dir ./$dir \
               --scale-interval $scale_interval \
               --scale-threshold $2 \
               --plot-title  $dir \
               --max-grad-norm 100000000000  \
               --saturation-log $dir/$dir.sat \
               --network-ratio $network_ratio  --reward-scaling $r
