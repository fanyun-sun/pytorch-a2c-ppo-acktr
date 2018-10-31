# pytorch-a2c-ppo-acktr

This repo contains the A2C code for the paper [ANS: Adaptive Network Scaling for Deep Rectifier Reinforcement Learning Models](https://arxiv.org/abs/1809.02112). It also implements [Pop-Art](https://arxiv.org/abs/1809.04474).
Largely based on [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr). Please refer to that for introductions or requirements. 

## Training

Start a `Visdom` server with `python -m visdom.server`, it will serve `http://localhost:8097/` by default.

### Example Usage

```
python main.py --env-name $env  \
               --hidden_size $network_size \
               --adaptive-interval $adaptive_interval \
               --tolerance $tolerance \
               --log-dir ./$dir \
               --plot-title  $dir \
               --log-interval 1000 --seed $seed 
```

### Important Arguments explained

`--pop-art`: whether to use pop-art or not.

`--adaptive-interval`: interval for adaptive reward scaling.

`--tolerance`: tolerance for adaptive reward scaling.

`--plot-title`: plot title for visdom.

`--log-interval`: interval for logging.

please refer to `arguments.py` for comprehensive explanation of the parameters.
