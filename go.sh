#!/bin/bash -ex

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/willy/.mujoco/mjpro150/bin
python main.py --env-name "HalfCheetah-v2" --num-stack 1 --num-frames 100000000 --max-grad-norm 100000000000 --log-dir ./tmpp --scale-interval 2000

