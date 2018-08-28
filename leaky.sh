#!/bin/bash -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/willy/.mujoco/mjpro150/bin

env=$1
network_size=64
scale_interval=1000000000000000
scale_threshold=2.
network_ratio=.5

reward=$2

seed=1
dir=$env-network_$network_size-network_ratio_$network_ratio-reward_scaling-$reward-seed_$seed-leaky

python main.py --env-name "$env"  \
               --num-stack 1  \
               --hidden_size $network_size \
               --log-dir ./$dir \
               --scale-interval $scale_interval \
               --scale-threshold $scale_threshold \
               --plot-title  $dir \
               --saturation-log $dir/$dir.sat \
               --network-ratio $network_ratio --adaptive-interval 1000000000000000 --leaky --reward-scaling $reward

#seed=2
#dir=$env-network_$network_size-network_ratio_$network_ratio-reward_scaling-$reward-seed_$seed-leaky
#python main.py --env-name "$env"  \
               #--num-stack 1  \
               #--hidden_size $network_size \
               #--log-dir ./$dir \
               #--scale-interval $scale_interval \
               #--scale-threshold $scale_threshold \
               #--plot-title  $dir \
               #--saturation-log $dir/$dir.sat \
               #--network-ratio $network_ratio --adaptive-interval 1000000000000000 --leaky --reward-scaling $reward --seed $seed

#seed=3
#dir=$env-network_$network_size-network_ratio_$network_ratio-reward_scaling-$reward-seed_$seed-leaky
#python main.py --env-name "$env"  \
               #--num-stack 1  \
               #--hidden_size $network_size \
               #--log-dir ./$dir \
               #--scale-interval $scale_interval \
               #--scale-threshold $scale_threshold \
               #--plot-title  $dir \
               #--saturation-log $dir/$dir.sat \
               #--network-ratio $network_ratio --adaptive-interval 1000000000000000 --leaky --reward-scaling $reward --seed $seed
