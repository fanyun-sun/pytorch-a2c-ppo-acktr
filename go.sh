#!/bin/bash -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/willy/.mujoco/mjpro150/bin

env=$1
network_size=64
scale_interval=1000000000000000
scale_threshold=2.
network_ratio=1.
r=$2
seed=$3

dir=$env-network_$network_size-network_ratio_$network_ratio-reward_scaling-$r-seed_$seed

python main.py --env-name "$env"  \
               --num-stack 1  \
               --hidden_size $network_size \
               --log-dir ./$dir \
               --scale-interval $scale_interval \
               --scale-threshold $2 \
               --plot-title  $dir \
               --saturation-log $dir/$dir.sat \
               --network-ratio $network_ratio  --reward-scaling $r --seed $seed
