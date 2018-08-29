#!/bin/bash -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/willy/.mujoco/mjpro150/bin

env=$1

network_size=64
scale_interval=1000000000000000
scale_threshold=2.
network_ratio=1.

adaptive_interval=1000000000000000000
lr=$2

dir=$env-network_$network_size-network_ratio_$network_ratio-pop_art-lr_$lr

python main.py --env-name "$env"  \
               --num-stack 1  \
               --hidden_size $network_size \
               --log-dir ./$dir \
               --scale-interval $scale_interval \
               --scale-threshold $scale_threshold \
               --plot-title  $dir \
               --saturation-log $dir/$dir.sat \
               --network-ratio $network_ratio \
               --adaptive-interval $adaptive_interval \
               --log-interval 1000 --pop-art --lr $lr
