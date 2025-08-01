# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import math
import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import random_quat
from .legged_robot_config import LeggedRobotCfg, USING_HYBRID
from rsl_rl.datasets.motion_loader import AMPLoader

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        # 1. 确定 各地形的 起止索引
        self.flat_start_idx = 0
        self.flat_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:1]))
        self.rough_start_idx = self.flat_end_idx
        self.rough_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:2]))
        self.smoothslope_start_idx = self.rough_end_idx
        self.smoothslope_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:3]))
        self.roughslope_start_idx = self.smoothslope_end_idx
        self.roughslope_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:4]))
        self.stairsup_start_idx = self.roughslope_end_idx
        self.stairsup_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:5]))
        self.stairsdown_start_idx = self.stairsup_end_idx
        self.stairsdown_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:6]))
        self.discreteobstacles_start_idx = self.stairsdown_end_idx
        self.discreteobstacles_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:7]))
        self.steppingstones_start_idx = self.discreteobstacles_end_idx
        self.steppingstones_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:8]))
        self.pit_start_idx = self.steppingstones_end_idx
        self.pit_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:9]))
        self.gap_start_idx = self.pit_end_idx
        self.gap_end_idx = self.cfg.env.num_envs

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False

        # 2. ------ 初始化 ------
        self.init_done = False
        # 2.1 初始化RL训练中每个episode的总步数（1000步）、域随机化中施加推力的步数间隔（800步）、obs的scale、rewars的scale、commands的范围
        self._parse_cfg(self.cfg)
        # 2.2 调用父类 BaseTask 的初始化：
        #   获取 env_cfg 中的 envs个数、obs维度等
        #   创建 envs, sim, viewer
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations  # 45
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs  # 45 + 3 + 3 + 187
        self.history_length = int(self.num_obs / self.num_one_step_obs)  # 45 * 6 / 45 = 6

        # 2.4 设置观察视角
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # 2.5 创建存储 仿真 state / obs / action 的 tensor
        self._init_buffers()
        # 2.6 将使用的 奖励函数 存放到 self.reward_functions 中，并为每个奖励函数 创建一个(num_env,)的tensor，存储在episode中每个env的奖励累计值
        self._prepare_reward_function()
        self.init_done = True
        # ------ 初始化完成 ------

        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 1. 当前env_step的 actions，裁剪到 [-100.0, 100.0]
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 2. 计算延迟后的 actions（域随机化）
        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)  # (num_envs, 4, 12)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)  # 每个 env 延迟的步数 [0, 4)
        if self.cfg.domain_rand.delay:
            # 计算延迟的4个actions（ < 延迟步数，则为上一时刻的actions，>= 延迟步数，则为最新时刻的actions）
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)

        # 3. 渲染（非headless模式）
        self.render()

        # 4. 执行一个 env_step（包含 4 个sim_step，即依次执行延迟后的4个actions）
        for _ in range(self.cfg.control.decimation):
            # 从 actions 计算 扭矩 (num_envs, 12)
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            # 应用 该扭矩 到 仿真环境
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)  # 执行物理仿真
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)  # 获取仿真结果
            self.gym.refresh_dof_state_tensor(self.sim)  # 更新 关节状态

        # 5. 执行 4个物理仿真步后的 操作
        # (1) 更新机器人的姿态
        # (2) 计算高度场
        # (3) 给base施加干扰
        # (4) 计算 奖励
        # (5) 计算 新的观测
        # (6) 重置某些env
        # (7) 更新上一env_step的数据（action、关节位置、关节速度、扭矩、base的线速度和角速度）
        #   返回： 需要重置的env的 ID (num_envs_,) 以及这些env的特权观测 (num_envs_, 45+3+3+187)
        # termination_ids, termination_priveleged_obs = self.post_physics_step()
        termination_ids, termination_priveleged_obs, terminal_amp_states = self.post_physics_step()

        # 6. 裁剪 观测obs_buf 到 [-100., 100.]
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # 7. 裁剪 特权观测privileged_obs_buf 到 [-100., 100.]
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if USING_HYBRID:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs, terminal_amp_states
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """
        更新机器人的姿态，计算高度场，给base施加干扰，计算 奖励、观测，重置某些env，更新上一env_step的数据（action、关节位置、关节速度、扭矩、base的线速度和角速度）

        Returns:
            env_ids: 需要重置的env的 ID (num_envs_,)
            terminal_amp_states: 需要重置的 env 的 AMP观测 (num_envs_, 30)
        """
        # 1. 从 Isaac Gym 仿真器中刷新各种状态张量，确保数据是最新的
        self.gym.refresh_actor_root_state_tensor(self.sim)   # 刷新 base的状态 张量
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新 关节接触力 张量
        self.gym.refresh_force_sensor_tensor(self.sim)       # 刷新 力传感器 张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)   # 刷新 刚体状态 张量

        # 2. 增加 当前回合的 步数计数器 和 通用步数计数器
        self.episode_length_buf += 1   # 当前回合的env_step数 +1
        self.common_step_counter += 1  # env_step数 +1

        # 3. 更新机器人的姿态、速度和重力投影信息
        self.base_quat[:] = self.root_states[:, 3:7]  # 更新机器人 base 的旋转四元数（世界坐标系）
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])   # 更新机器人 base 的 线速度（body坐标系）
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 更新机器人 base 的 角速度（body坐标系）
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 更新投影到机器人坐标系的 重力向量（body坐标系）

        # 四足的 位置 和 线速度（世界坐标系）
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        # 四足的 接触力
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        # 原代码在计算奖励前调用 _post_physics_step_callback()，这可能不合理。例如，当前动作遵循当前命令，而 _post_physics_step_callback() 可能会重新采样命令，导致奖励较低。
        # 4. 每500个env_step重新采样这些env的commands，
        # 计算地形高度场，
        # 给base施加 水平速度干扰
        # 给base施加 力干扰
        self._post_physics_step_callback()

        # 5. 检查环境是否需要 重置
        self.check_termination()

        # 6. 计算 奖励
        self.compute_reward()

        # 7. 重置某些 env
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 获取需要重置的 env ID
        # 获取需要重置的 env 的特权观测 (num_envs_, 45+3+3+187)
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)  # 重置这些 env

        # 8. 计算 观测
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # 9. 更新上一env_step的 actions、关节位置、关节速度、扭矩、base的线速度和角速度
        self.disturbance[:, :, :] = 0.0  # 给各刚体的扰动力 清零
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        # return env_ids, termination_privileged_obs
        return env_ids, termination_privileged_obs, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_counts = {}
        # (1) 触发终止部位的接触力 > 1N，则需要重置 (num_envs,)
        contact_force_cond = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf = contact_force_cond
        termination_counts["contact_force"] = (contact_force_cond.sum().item() / self.num_envs) * 100
        # print(f'[legged_robot] termination_counts contact_force (%): {termination_counts["contact_force"]}]')

        # (2) episode步数 > 1000
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        termination_counts["time_out"] = (self.time_out_buf.sum().item() / self.num_envs) * 100
        # print(f'[legged_robot] termination_counts time_out (%): {termination_counts["time_out"]}]')

        # (3) base速度 与 命令速度（因摔倒恢复训练，需关闭这个）
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "base_vel_violate_commands", False):
            vel_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
            self.vel_violate = ((vel_error > 2) & (self.commands[:, 0] < 0.)) | ((vel_error < -2) & (self.commands[:, 0] > 0.))
            self.vel_violate *= (self.terrain_levels > 3)
            self.reset_buf |= self.vel_violate
            termination_counts["vel_violate"] = (self.vel_violate.sum().item() / self.num_envs) * 100
            # print(f'[legged_robot] termination_counts vel_violate (%): {termination_counts["vel_violate"]}]')

        # (4) env走出地形的边界
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "out_of_border", False):
            out_border = self.terrain.in_terrain_range(self.root_states[:, :3], device=self.device).logical_not()
            self.reset_buf |= out_border
            termination_counts["out_border"] = (out_border.sum().item() / self.num_envs) * 100
            # print(f'[legged_robot] termination_counts out_border (%): {termination_counts["out_border"]}]')

        # (5) base的z方向线速度 < -5 （即跌落）# 或 重力投影 为 Z轴向上（因摔倒恢复训练，需关闭这个，避免刚重置就终止了）
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "fall_down", False):
            fall_down = (self.root_states[:, 9] < -5.)  #  | (self.projected_gravity[:, 2] > 0.)
            self.reset_buf |= fall_down
            termination_counts["fall_down"] = (fall_down.sum().item() / self.num_envs) * 100
            # print(f'[legged_robot] termination_counts fall_down (%): {termination_counts["fall_down"]}]')

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # 1. 更新地形课程（根据机器人表现调整地形难度）
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # 2. 更新 commands 课程（调整速度命令范围）
        # 避免每步都更新，因为最大命令对所有env是共享的
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
        
        # 重置关节、base状态
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        # 4. 为重置的 env 重新采样 commands
        self._resample_commands(env_ids)

        # 6. 重置各种缓冲区
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
         # 重新获取 域随机化 数据
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        # 重新获取 env的摩擦系数、弹性系数，并设置给env的各部位
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # 记录episode信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # 遍历每个奖励函数，计算对应的 这些重置的env在当前episode内的 (平均奖励值 / 0.02)的均值
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        # 记录课程信息
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            # 记录当前命令范围
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        #  发送超时信息给算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.  # (num_envs,)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew  # 对应env的 所有奖励之和 (num_envs,)
            self.episode_sums[name] += rew  # 对应奖励函数 在当前episode内的 对应env的 奖励之和 (num_envs,)
        if self.cfg.rewards.only_positive_rewards:  # 默认不执行
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # 在裁剪至0后，重新赋值终止的惩罚
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]  # (num_envs,)
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,  # * [2, 2, 0.25]
                                    self.base_ang_vel  * self.obs_scales.ang_vel,  # 0.25
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 1.0
                                    self.dof_vel * self.obs_scales.dof_vel,  # 0.05
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)  # base线速度 * 2.0, 给base施加的随机扰动力(xyz方向)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)  # 6 steps
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def get_amp_observations(self):
        joint_pos = self.dof_pos
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # z_pos = self.root_states[:, 2:3]
        # if (self.cfg.terrain.measure_heights):
        #     z_pos = z_pos - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)

    def get_current_obs(self):
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations (num_envs, 45+3+3+187)
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # 创建 terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        # 创建 agents
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # 为每个env生成一个随机摩擦系数，并将同一个env的所有刚体部位的摩擦系数 都设置为相同的数值
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device=self.device)
                # 为每个env生成一个随机摩擦数 (num_env, 1)
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        # 为每个env生成一个随机弹性系数，并将同一个env的所有刚体部位的弹性系数 都设置为相同的数值
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1), device=self.device)
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """
         存储、处理、随机 关节属性，包括：位置限制、速度限制、力矩限制（env 创建期间被调用）

        Args:
            props (numpy.array): 每个关节的属性数组，包含 位置/速度/力矩
            env_id (int): 当前环境ID，用于判断是否需要初始化限制参数

        Returns:
            [numpy.array]: 原始属性（未修改）
        """
        # 只在第一个环境初始化时设置关节限制
        if env_id==0:
            # 初始化存储关节限制的张量
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)  # (num_dof, 2)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # (num_dof,)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)   # (num_dof,)
            # 遍历每个关节属性
            for i in range(len(props)):
                # 存储原始关节限制
                self.dof_pos_limits[i, 0] = props["lower"][i].item()  # 最小 位置 限制
                self.dof_pos_limits[i, 1] = props["upper"][i].item()  # 最大 位置 限制
                self.dof_vel_limits[i] = props["velocity"][i].item()  # 最大 速度 限制
                self.torque_limits[i] = props["effort"][i].item()  # 最大 力矩 限制

                # 计算软限制（比硬件限制更宽松的范围）
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 中间值
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 范围
                # 根据配置设置软限制范围（self.cfg.rewards.soft_dof_pos_limit 通常为0.9）
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props  # 返回原始属性（未修改）

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # 随机更改base的质量
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]

        # 随机更改base的质心
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        # 随机更改env各刚体部位（除了base）的质量
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 1. envs中 当其episode的 env_step数达到500步，则重新采样commands
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # 2. 根据目标航向角偏差，计算commands的角速度
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)  # 当前base的前进方向（世界坐标系）(num_envs, 3)
            heading = torch.atan2(forward[:, 1], forward[:, 0])  # 当前base的前进方向的 航向角
            # 命令的角速度 = 0.5 * (目标航向 - 当前航向)==> 裁剪到[-2, 2]
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)

        # 3. 计算采样点的高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 4. 每16/0.02s个 env_step，给base在水平方向施加一个速度
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        # 5. 每8个 env_step，给base施加一个随机的力
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # 重新采样env_ids的commands
        self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # 处理高速 env 的commands（前 20%）
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)
        # set y commands of high vel envs to zero
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions = self.motor_strength * actions
        # 1. 根据输入的 actions，计算关节目标位置 = default_dof_pos + actions * 0.5
        actions_scaled = actions * self.cfg.control.action_scale  # actions * 0.5
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_reduction  # hip关节的再 * 1.0
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        # 2. 根据控制类型计算扭矩
        control_type = self.cfg.control.control_type
        if control_type=="P":  # 位置控制模式
            # 扭矩 = P增益 * 域随机化系数 * (关节目标位置 - 关节当前位置) + D增益 * 域随机化系数 * 关节当前速度
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":  # 速度控制模式
            # 扭矩 = P增益 * (目标速度 - 当前速度) + D增益 * 加速度              其中，加速度 = (当前速度 - 上一时刻速度) / 物理时间步长
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type=="T":  # 扭矩控制模式
            # 直接使用 缩放后的action 作为 扭矩
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)  # 裁剪到扭矩限制范围内

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if getattr(self.cfg.domain_rand, "dof_init_pos_ratio_range", None) is not None:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.dof_init_pos_ratio_range[0],
                self.cfg.domain_rand.dof_init_pos_ratio_range[1],
                (len(env_ids), self.num_dof),
                device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos

        if getattr(self.cfg.domain_rand, "randomize_dof_vel", False):
            dof_vel_range = getattr(self.cfg.domain_rand, "init_dof_vel_range", [-1.0, 1.0])
            self.dof_vel[env_ids] = torch.rand_like(self.dof_vel[env_ids]) * abs(dof_vel_range[1] - dof_vel_range[0]) + min(dof_vel_range)
        else:
            self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if hasattr(self.cfg.domain_rand, "base_init_pos_range"):
                self.root_states[env_ids, 0:1] += torch_rand_float(
                    *self.cfg.domain_rand.base_init_pos_range["x"],
                    (len(env_ids), 1),
                    device=self.device
                )
                self.root_states[env_ids, 1:2] += torch_rand_float(
                    *self.cfg.domain_rand.base_init_pos_range["y"],
                    (len(env_ids), 1),
                    device=self.device
                )
                # random height
                self.root_states[env_ids, 2:3] += torch_rand_float(
                    *self.cfg.domain_rand.base_init_pos_range["z"],
                    (len(env_ids), 1),
                    device=self.device
                )
            else:  # 默认x,y方向为 [-1, 1], z方向为 0
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # base rotation (roll and pitch)
        if hasattr(self.cfg.domain_rand, "base_init_rot_range"):
            base_roll = torch_rand_float(
                *self.cfg.domain_rand.base_init_rot_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_pitch = torch_rand_float(
                *self.cfg.domain_rand.base_init_rot_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_yaw = torch_rand_float(
                *self.cfg.domain_rand.base_init_rot_range.get("yaw", [-np.pi, np.pi]),
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_quat = quat_from_euler_xyz(base_roll, base_pitch, base_yaw)
            self.root_states[env_ids, 3:7] = base_quat

        # base velocities
        if getattr(self.cfg.domain_rand, "base_init_vel_range", None) is not None:
            base_vel_range = self.cfg.domain_rand.base_init_vel_range
        else:
            base_vel_range = (-0.5, 0.5)
        if isinstance(base_vel_range, (tuple, list)):
            self.root_states[env_ids, 7:13] = torch_rand_float(
                *base_vel_range,
                (len(env_ids), 6),
                device=self.device
            ) # [7:10]: lin vel, [10:13]: ang vel
        elif isinstance(base_vel_range, dict):
            self.root_states[env_ids, 7:8] = torch_rand_float(
                *base_vel_range["x"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 8:9] = torch_rand_float(
                *base_vel_range["y"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 9:10] = torch_rand_float(
                *base_vel_range["z"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 10:11] = torch_rand_float(
                *base_vel_range["roll"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 11:12] = torch_rand_float(
                *base_vel_range["pitch"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 12:13] = torch_rand_float(
                *base_vel_range["yaw"],
                (len(env_ids), 1),
                device=self.device
            )
        else:
            raise NameError(f"Unknown base_vel_range type: {type(base_vel_range)}")

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. (瞬时的)
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy  # 获取推动env的最大线速度 [1m/s]
        # 给 base 的xy方向线速度 上再添加随机 速度
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_reward_curriculum(self, current_iter):
        for i in range(len(self.cfg.rewards.reward_curriculum_schedule)):
            percentage = (current_iter - self.cfg.rewards.reward_curriculum_schedule[i][0]) / \
                         (self.cfg.rewards.reward_curriculum_schedule[i][1] - self.cfg.rewards.reward_curriculum_schedule[i][0])
            percentage = max(min(percentage, 1), 0)
            self.reward_curriculum_coef[i] = (1 - percentage) * self.cfg.rewards.reward_curriculum_schedule[i][2] + \
                                          percentage * self.cfg.rewards.reward_curriculum_schedule[i][3]

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        # [-30, 30] N
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance  # 给 base 添加随机扰动力
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            # [-2, 2] ==> [-1.0, 1.5]
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -self.cfg.commands.max_backward_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., self.cfg.commands.max_forward_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.1, -self.cfg.commands.max_lat_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.1, 0., self.cfg.commands.max_lat_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)
        else:
            noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        #noise_vec[232:] = 0
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # 从Isaac Gym仿真器中获取各种 state tensor
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # base的状态
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 关节状态
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 每个刚体的接触力
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 刚体（包含base和各部件的）的状态

        # 刷新这些张量以确保数据最新
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 将获取的原始张量包装成PyTorch张量
        # base的状态 (num_envs, 13)，[0:3] base的位置, [3:7] base的旋转四元数，[7:10] base的线速度，[10:13] base的角速度
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # 关节状态 (num_envs * num_dof, 2)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 当前 关节位置 (num_env, 12, 1)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 当前 关节速度 (num_env, 12, 1)
        self.base_quat = self.root_states[:, 3:7]  # base 的旋转四元数（世界坐标系）

        # 刚体（包含base和各部件的）的状态 (num_env, num_bodies 17, 3)，[0:3]是位置，[7:10]是线速度
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # 四足的 位置 和 线速度（世界坐标系）
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        # 存储每个刚体在xyz方向的接触力，(num_envs, num_bodies 17, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        # 初始化计数器、额外数据、重力向量等
        self.common_step_counter = 0  # 步数计数器
        self.extras = {}  # 额外数据字典
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))    # 机器人的前进方向（base坐标系）
        # 初始化 torques
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # (num_envs, 12)
        # 初始化 PD增益
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # P增益 (num_envs, 12)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # D增益 (num_envs, 12)
        # 初始化 actions，四肢的关节位置（按腿的顺序：FL, FR, RL, RR），(num_envs, 12)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])  # base的 线速度 + 角速度
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # motor_strength
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if getattr(self.cfg.domain_rand, "randomize_motor_strength", False):
            mtr_rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        
        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # 给各刚体施加的 干扰力
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # 从所有奖励函数中 去除 env_cfg 中 rewards.scales 为 0 的项
        # 非0的各奖励函数的 scales * self.dt (0.02)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)  # reward 名称列表，没有 _reward_前缀
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # 对应奖励函数 在当前episode内的 对应env的 奖励之和（为 每个scale非0的 奖励函数 创建一个 (num_env,) tensor）
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        # 总地形网格（水平方向）的 个数 (10 * 80 + 2 * 150, 20 * 80 + 2 * 150)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # 包含刚体属性的列表，即包含机器狗的每个部位，[0]通常表示base
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        # ['base', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot', 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot']
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins() # 获取每个env初始化时在地形中的位置 = 对应子地形的中心位置 (num_envs, 3)
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        # for domain randomization
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # 获取给env的质量 加减的范围
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        # 获取给base的 位置（xyz）加减的范围
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)

        # 创建每一个env
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()  # 该env的初始位置
            # (1) env初始位置随机 xy方向 随机加减1
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # (2) 为每个env生成一个随机摩擦系数、弹性系数
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # (3) 计算关节的属性限制（位置、速度、力矩）
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            # (4) 随机更改env的 base质量、质心偏移、其他刚体部位质量
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass

            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # (num_envs, 8)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            # 在 [0, max_init_level + 1] 范围中随机生成每个env的 初始地形等级 (num_env,)
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            # env 平均分布在每列地形中 的编号 (num_envs / num_cols,)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            # 各子地形的中心位置 (num_rows, num_cols, 3)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            # 每个env初始化时在地形中的位置 = 对应子地形的中心位置 (num_envs, 3)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # policy步长（env_step步长） = 0.02 = 4 * 0.005(物理仿真步长)
        self.obs_scales = self.cfg.normalization.obs_scales  # 各观测值的 缩放系数 (2.0, 0.25, 1.0, 0.05, 5.0)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)  # 各奖励项的 缩放系数
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)  # 各 command 的 范围
        # 非网格地形，则禁用 课程学习
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s  # 20s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # 20s / 0.02 = 1000 steps

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 16s / 0.02 = 800 steps

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()  # 9 * 7 = 63
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    # ------------ reward functions ------------
    def _reward_tracking_lin_vel(self):
        # 奖励 跟踪 commands 中XY方向的 线速度
        small_commands = torch.norm(self.commands[:, :2], dim=1) < 0.1
        track_commands = self.commands[:, :2] * (~small_commands.unsqueeze(-1))
        lin_vel_error = torch.sum(
            torch.square(track_commands - self.base_lin_vel[:, :2]),
            dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # 奖励 跟踪 commands 中yaw方向角速度
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # 奖励 四足的空中时间接近0.5s (2Hz)
        # 需过滤接触力信号，因为PhysX引擎在复杂地形上接触力检测不可靠
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # 检测z轴力 > 1N 的接触
        contact_filt = torch.logical_or(contact, self.last_contacts)  # 当前帧和上一帧的 有1次触地即可
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt  # 只考虑从空中首次触地的情况
        self.feet_air_time += self.dt  # 累加 policy 步长（0.02s）
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # 仅奖励第一次触地，且计算与目标时间0.5s的偏差奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # commands线速度 > 0.1 时才奖励
        self.feet_air_time *= ~contact_filt  # 当前帧 触地的足 空中时间清0
        return rew_airTime

    def _reward_upward(self):
        # 奖励 重力投影向下
        return 1 - self.projected_gravity[:, 2]

    def _reward_has_contact(self):
        # 奖励 速度<0.1时 的 四足接触力
        contact_filt = 1. * self.contact_filt
        return (torch.norm(self.commands[:, :2], dim=1) < 0.1) * torch.sum(contact_filt, dim=-1) / 4

    # ------------ penalty functions ------------
    def _reward_lin_vel_z(self):
        # 惩罚 base 的 Z 轴线速度（防止跳跃）
        return torch.square(self.base_lin_vel[:, 2])
    def _reward_lin_vel_z_up(self):
        return torch.square(self.base_lin_vel[:, 2]) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)
    
    def _reward_ang_vel_xy(self):
        # 惩罚 base 的 XY 轴角速度（roll, pitch, 防止翻滚）
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    def _reward_ang_vel_xy_up(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)
    
    def _reward_orientation(self):
        # 惩罚 base 非水平姿态
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    def _reward_orientation_up(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_base_height(self):
        # 惩罚 base 偏离目标高度
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    def _reward_base_height_up(self):
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    # --- dof velocity ---
    def _reward_dof_vel(self):
        # 惩罚 关节速度
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # 惩罚 关节加速度
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_vel_limits(self):
        # 惩罚 关节速度接近极限
        # 裁剪至 max error = 每个关节 1 rad/s，以避免 巨大惩罚
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # --- dof position ---
    def _reward_dof_pos_dif(self):
        # 惩罚 关节位置 的变化
        return torch.sum(torch.square(self.last_dof_pos - self.dof_pos), dim=1)

    def _reward_dof_pos_limits(self):
        # 惩罚 关节位置接近极限
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # --- actions ---
    def _reward_action_rate(self):
        # 惩罚 action 的变化（使机器人运动更加平滑连续）
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # 惩罚 action 的二阶平滑性（使动作更加平缓）
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    # --- torques ---
    def _reward_torques(self):
        # 惩罚 关节扭矩过大（防止关节过热或损坏）
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_torques_distribution(self):
        # 惩罚 关节扭矩分布不均
        return torch.var(torch.abs(self.torques), dim=1)

    def _reward_torques_dif(self):
        # 惩罚 关节扭矩的变化
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_torque_limits(self):
        # 惩罚 关节扭矩接近极限
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    # --- power ---
    def _reward_joint_power(self):
        # 惩罚 高功率
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_power(self):
        # 惩罚 关节功率消耗（扭矩 * 关节速度）
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_power_distribution(self):
        # 惩罚 关节功率消耗分布不均
        return torch.var(torch.abs(self.torques * self.dof_vel), dim=1)

    # --- collision, termination
    def _reward_collision(self):
        # 惩罚 指定关节的碰撞
        # 当指定关节接触力的 模 > 0.1N，则判定发生碰撞，计为 1
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    def _reward_collision_up(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    # --- feet contact ---
    def _reward_feet_contact_forces(self):
        # 惩罚 四足接触力过大（需<100）
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_feet_stumble(self):
        # 惩罚 四足接触到垂直表面
        # 判定条件： XY方向 足部接触力 与 Z轴接触力 之比 > 5
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        rew = rew * (self.terrain_levels > 3)
        rew = rew.float()
        stumble_reward = torch.zeros_like(rew)
        stumble_reward[self.stairsup_start_idx: self.stairsup_end_idx] = rew[self.stairsup_start_idx: self.stairsup_end_idx]
        stumble_reward[self.pit_start_idx: self.gap_end_idx] = rew[self.pit_start_idx: self.gap_end_idx]
        return stumble_reward
    def _reward_feet_stumble_up(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        rew = rew * (self.terrain_levels > 3)
        rew = rew.float()
        stumble_reward = torch.zeros_like(rew)
        stumble_reward[self.stairsup_start_idx: self.stairsup_end_idx] = rew[self.stairsup_start_idx: self.stairsup_end_idx]
        stumble_reward[self.pit_start_idx: self.gap_end_idx] = rew[self.pit_start_idx: self.gap_end_idx]
        return stumble_reward * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_feet_slide(self):
        # 惩罚 触地时 四足相对base的速度（避免滑动）
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)  # 当前四足相对base的 线速度（世界坐标系）
        # 当前四足相对base的 线速度（body坐标系）
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        # 四足相对base的 线速度 的模
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(self.contact_filt * foot_leteral_vel, dim=1)
    def _reward_feet_slide_up(self):
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(self.contact_filt * foot_leteral_vel, dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_feet_contact_forces(self):
        # 惩罚 四足的接触力 > 100N
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_mirror(self):
        # 惩罚 斜对称腿的关节位置偏差
        diff1 = torch.sum(torch.square(self.dof_pos[:, [1, 2]] - self.dof_pos[:, [10, 11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:, [4, 5]] - self.dof_pos[:, [7, 8]]),dim=-1)
        return 0.5 * (diff1 + diff2)
    def _reward_feet_mirror_up(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:, [1, 2]] - self.dof_pos[:, [10, 11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:, [4, 5]] - self.dof_pos[:, [7, 8]]),dim=-1)
        return 0.5 * (diff1 + diff2) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    # --- stand, stuck ---
    def _reward_stand_still(self):
        # 惩罚 commands 速度接近0（<0.1 m/s）时的 关节位置与默认关节位置的 偏差
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_stand_nice(self):
        # 惩罚 commands 速度接近0（<0.1 m/s）且 重力投影向下时 的 关节位置与默认关节位置的 偏差
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1) * (1 - self.projected_gravity[:,2])

    def _reward_stuck(self):
        # 惩罚 卡住
        # 判定是否卡住：base 的 X方向线速度 < 0.1 m/s，但 commands 的 X方向线速度 > 0.1 m/s
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)

    # --- joint pose deviation ---
    def _reward_hip_action_magnitude(self):
        # 限制 action 中的 髋关节hip（0,3,6,9）动作幅度（防止 > 1.0）
        return torch.sum(torch.square(torch.maximum(torch.abs(self.actions[:, [0, 3, 6, 9]]) - 1.0,
                                                    torch.zeros_like(self.actions[:, [0, 3, 6, 9]]))), dim=1)

    def _reward_hip_pos(self):
        # 惩罚 髋关节hip（0,3,6,9）与默认位置的偏差
        return torch.sum(torch.abs(self.dof_pos[:, [0,3,6,9]] - self.default_dof_pos[:, [0,3,6,9]]), dim=1)
    def _reward_hip_pos_up(self):
        return (torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
                * torch.clamp(-self.projected_gravity[:, 2], 0, 1))

    def _reward_thigh_pose(self):
        return torch.sum(torch.abs(self.dof_pos[:, [1,4,7,10]] - self.default_dof_pos[:, [1,4,7,10]]), dim=1)
    def _reward_thigh_pose_up(self):
        return (torch.sum(torch.abs(self.dof_pos[:, [1,4,7,10]] - self.default_dof_pos[:, [1,4,7,10]]), dim=1)
                * torch.clamp(-self.projected_gravity[:, 2], 0, 1))

    def _reward_calf_pose(self):
        return torch.sum(torch.abs(self.dof_pos[:, [2,5,8,11]] - self.default_dof_pos[:, [2,5,8,11]]), dim=1)
    def _reward_calf_pose_up(self):
        return (torch.sum(torch.abs(self.dof_pos[:, [2,5,8,11]] - self.default_dof_pos[:, [2,5,8,11]]), dim=1)
                * torch.clamp(-self.projected_gravity[:, 2], 0, 1))

    # --- 四足离地高度 ---
    def _reward_foot_clearance_base(self):
        # 惩罚 大速度下 四足抬脚距base的高度 偏离目标距离 （-0.2 m）（摔倒时不计算）
        # 当前四足相对base的 位置 和 线速度（世界坐标系）
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        # 当前四足相对base的 位置 和 线速度（body坐标系）
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)  # (num_envs, 4, 3)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])

        # 四足相对base的高度 距 目标高度 的误差（平方误差）
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.foot_height_target_base).view(self.num_envs, -1)
        # 四足相对base的 线速度 的模
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    def _reward_foot_clearance_base_up(self):
        # 惩罚 大速度下 四足抬脚距base的高度 偏离目标距离 （-0.2 m）（摔倒时不计算）
        # 当前四足相对base的 位置 和 线速度（世界坐标系）
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        # 当前四足相对base的 位置 和 线速度（body坐标系）
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)  # (num_envs, 4, 3)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])

        # 四足相对base的高度 距 目标高度 的误差（平方误差）
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.foot_height_target_base).view(self.num_envs, -1)
        # 四足相对base的 线速度 的模
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_foot_clearance_terrain(self):
        # 惩罚 大速度下 四足的抬脚高度 需接近 离地目标高度（0.15m）
        if self.cfg.terrain.mesh_type == 'plane':
            foot_heights = self.feet_pos[:, :, 2]  # 四足的 高度 (num_envs, 4, 1)
        else:
            points = self.feet_pos  # 四足的位置 (num_envs, 4, 3)

            # 测量 四足位置下方的 地形高度
            points += self.terrain.cfg.border_size  # + 边界的偏移 25
            points = (points / self.terrain.cfg.horizontal_scale).long()  # / 0.1，归一化到地形网格坐标
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            heights1 = self.height_samples[px, py]
            heights2 = self.height_samples[px + 1, py]
            heights3 = self.height_samples[px, py + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale  # 地形高度 转换为 实际的米单位 (num_evns, 4)
            foot_heights = self.feet_pos[:, :, 2] - ground_heights  # 四足相对地形的 高度

        foot_lateral_vel = torch.norm(self.feet_vel[:, :, :2], dim = -1)
        # return torch.sum(foot_lateral_vel * torch.maximum(-foot_heights + self.cfg.rewards.foot_height_target, torch.zeros_like(foot_heights)), dim = -1)
        return torch.sum(foot_lateral_vel * torch.square(foot_heights - self.cfg.rewards.foot_height_target_terrain), dim = -1)
    def _reward_foot_clearance_terrain_up(self):
        # 惩罚 大速度下 四足的抬脚高度 需接近 离地目标高度（0.15m）
        if self.cfg.terrain.mesh_type == 'plane':
            foot_heights = self.feet_pos[:, :, 2]  # 四足的 高度 (num_envs, 4, 1)
        else:
            points = self.feet_pos  # 四足的位置 (num_envs, 4, 3)

            # 测量 四足位置下方的 地形高度
            points += self.terrain.cfg.border_size  # + 边界的偏移 25
            points = (points / self.terrain.cfg.horizontal_scale).long()  # / 0.1，归一化到地形网格坐标
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            heights1 = self.height_samples[px, py]
            heights2 = self.height_samples[px + 1, py]
            heights3 = self.height_samples[px, py + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale  # 地形高度 转换为 实际的米单位 (num_evns, 4)
            foot_heights = self.feet_pos[:, :, 2] - ground_heights  # 四足相对地形的 高度

        foot_lateral_vel = torch.norm(self.feet_vel[:, :, :2], dim = -1)
        # return torch.sum(foot_lateral_vel * torch.maximum(-foot_heights + self.cfg.rewards.foot_height_target, torch.zeros_like(foot_heights)), dim = -1)
        return torch.sum(foot_lateral_vel * torch.square(foot_heights - self.cfg.rewards.foot_height_target_terrain), dim = -1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)