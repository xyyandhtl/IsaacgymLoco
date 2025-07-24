## HimLoco

- forked from: https://github.com/InternRobotics/HIMLoco
- him paper: https://arxiv.org/abs/2404.14405
- hinf paper: https://arxiv.org/abs/2304.08485 (code to be released)
- amp from: https://github.com/bytedance/WMP


### Installation
1. Create an environment and install PyTorch:

2. Install Isaac Gym:
  - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  - `cd isaacgym/python && pip install -e .`

3. Clone this repository.
  - `cd HIMLoco`

4. Install HIMLoco.
  - `cd rsl_rl && pip install -e .`
  - `cd ../legged_gym && pip install -e .`

### Usage
1. Train a policy:
* flat terrain
  - `python legged_gym/legged_gym/scripts/train.py --task aliengo --headless`
  - `python legged_gym/legged_gym/scripts/train.py --task aliengo_recover --headless`
* stairs terrain
  - change the resume flat terrain log path in `legged_gym/legged_gym/envs/aliengo/aliengo_stairs_config.py` lines 192 `load_run = ...` and change `resume = True`
  - `python legged_gym/legged_gym/scripts/train.py --task aliengo_stairs --headless`
  
    or 
  - `python legged_gym/legged_gym/scripts/train --task aliengo_stairs --resume --load_run Jul29_14-35-18_ --headless`


* to use amp, modify [legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py), train scripts same with above
```python
USING_HYBRID = True
```

2. Play and export the latest policy:
   - `python legged_gym/legged_gym/scripts/play.py --task aliengo --load_run <run_name>`
   - `python legged_gym/legged_gym/scripts/play.py --task aliengo_stairs --load_run <run_name>`
