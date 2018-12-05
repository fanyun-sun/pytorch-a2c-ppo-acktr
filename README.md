# pytorch-a2c-ppo-acktr

This repo contains the A2C code for the paper [ANS: Adaptive Network Scaling for Deep Rectifier Reinforcement Learning Models](https://arxiv.org/abs/1809.02112). It also implements [Pop-Art](https://arxiv.org/abs/1809.04474).
Please refer to the repository [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) for detailed introductions or requirements. 

## Training

Start a `Visdom` server with `python -m visdom.server`, it will serve `http://localhost:8097/` by default.

### Example Usage and parameters

```
python main.py --env-name $env  \
               --hidden_size 64 \
               --adaptive-interval 100 \
               --tolerance 60 \
               --cdec .9 \
               --cinc 8 \
               --log-dir ./"$env"-log \
               --plot-title  $env \
               --log-interval 1000 
               --seed 1 
```

### Important Arguments explained

* `--pop-art`: whether to use pop-art or not.

* `--adaptive-interval`: interval for adaptive reward scaling.

* `--tolerance`: tolerance for adaptive reward scaling.

* `--plot-title`: plot title for visdom server.

* `--log-interval`: interval for logging.

please refer to `arguments.py` for comprehensive explanation of all arguments.
