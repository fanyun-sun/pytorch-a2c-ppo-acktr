#!/bin/bash -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/willy/.mujoco/mjpro150/bin

env=Hopper-v2
dir=$env-network_8-scale_thresh_.4

python main.py --env-name "$env"  \
               --num-stack 1  \
               --num-frames 100000000  \
               --max-grad-norm 100000000000  \
               --log-dir ./$dir \
               --scale-interval 10000000000000 \
               --scale-threshold .4 \
               --plot-title  $dir \
               --saturation-log $dir.sat 


python main.py --env-name "$env"  \
               --num-stack 1  \
               --num-frames 100000000  \
               --max-grad-norm 100000000000  \
               --log-dir ./$dir \
               --scale-interval 5000 \
               --plot-title  $dir \
               --saturation-log $dir.sat 
