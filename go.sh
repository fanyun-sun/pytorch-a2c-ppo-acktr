#!/bin/bash -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/willy/.mujoco/mjpro150/bin

env=Hopper-v2

python main.py --env-name "$env"  \
               --num-stack 1  \
               --num-frames 100000000  \
               --max-grad-norm 100000000000  \
               --log-dir ./network_256 \
               --scale-interval 10000000000 \
               --plot-title  $env-network_256 \
               --saturation-log network_256.sat

