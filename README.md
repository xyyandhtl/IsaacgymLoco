## HimLoco

- forked from: https://github.com/InternRobotics/HIMLoco
- him paper: https://arxiv.org/abs/2404.14405
- hinf paper: https://arxiv.org/abs/2304.08485 (code to be released)
- amp from: https://github.com/bytedance/WMP

### Usage

```shell
python legged_gym/legged_gym/scripts/train --task aliengo --headless
python legged_gym/legged_gym/scripts/train --task aliengo_stairs --resume --load_run Jul29_14-35-18_ --headless
```

- to use amp, modify [legged_robot_config.py](legged_gym/legged_gym/envs/base/legged_robot_config.py), train scripts same with above
```python
USING_HYBRID = True
```

