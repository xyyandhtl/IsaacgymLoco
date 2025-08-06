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

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, HybridPPO
from rsl_rl.modules import HIMActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
# from rsl_rl.modules import DepthPredictor
import torch.optim as optim

import ruamel.yaml as yaml
import argparse
import pathlib
import sys

class HybridPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        # self.depth_predictor_cfg = train_cfg["depth_predictor"]
        self.device = device
        self.env = env
        # self.history_length = history_length
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs  # 45 + 3 + 3 + 187
        else:
            num_critic_obs = self.env.num_obs
        # if self.env.include_history_steps is not None:
        #     num_actor_obs = self.env.num_obs * self.env.include_history_steps
        # else:
        #     num_actor_obs = self.env.num_obs
        self.num_actor_obs = self.env.num_obs  # 45 * 6
        self.num_critic_obs = num_critic_obs

        actor_critic_class = eval(self.cfg["policy_class_name"])  # HIMActorCritic
        actor_critic: HIMActorCritic = actor_critic_class(self.env.num_obs,  # 45 * 6
                                                          num_critic_obs,  # 45 + 3 + 3 + 187
                                                          self.env.num_one_step_obs,  # 45
                                                          self.env.num_actions,  # 12
                                                          **self.policy_cfg).to(self.device)

        # build world model
        # self._build_world_model()

        # build depth predictor
        # self.depth_predictor = DepthPredictor().to(self._world_model.device)
        # self.depth_predictor_opt = optim.Adam(self.depth_predictor.parameters(), lr=self.depth_predictor_cfg["lr"],
        #                                       weight_decay=self.depth_predictor_cfg["weight_decay"])

        # self.history_dim = history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim-3) #exclude command
        # actor_critic = ActorCriticWMP(num_actor_obs=num_actor_obs,
        #                                   num_critic_obs=num_critic_obs,
        #                                   num_actions=self.env.num_actions,
        #                                   height_dim=self.env.height_dim,
        #                                   privileged_dim=self.env.privileged_dim,
        #                                   history_dim=self.history_dim,
        #                                   wm_feature_dim=self.wm_feature_dim,
        #                                   **self.policy_cfg).to(self.device)

        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"])  # HIMPPO
        min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        self.alg: HybridPPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                                        min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        # self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs],
        #                       [self.env.num_privileged_obs], [self.env.num_actions], self.history_dim, self.wm_feature_dim)
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if self.env.cfg.rewards.reward_curriculum:
                self.env.update_reward_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                # 进行 num_steps_per_env 次 env_step
                for i in range(self.num_steps_per_env):
                    # if (self.env.global_counter % self.wm_update_interval == 0):
                    #     # world model obs step
                    #     wm_embed = self._world_model.encoder(wm_obs)
                    #     wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                    #                                                        wm_obs["is_first"])
                    #     wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                    #     wm_is_first[:] = 0

                    # 1. 根据当前观测 计算actions正态分布 和 价值
                    # history = self.trajectory_history.flatten(1).to(self.device)
                    # actions = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature.to(self.env.device))
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    # 2. 在仿真环境中应用该actions，执行执行一个 env_step（包含 4 个sim_step）
                    # 获取：新观测、新特权观测、奖励buffer、重置buffer、额外信息、要重置env的ID、要重置env的特权观测
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, termination_privileged_obs, terminal_amp_states = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # 2.5. 处理terminal cases, amp和critic
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)

                    reset_env_ids = reset_env_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[reset_env_ids] = termination_privileged_obs.clone().detach()
                    # 3. 将当前env_step完成后的数据 记录到  rollout 中
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term, next_critic_obs)

                    # process trajectory history
                    # env_ids = dones.nonzero(as_tuple=False).flatten()
                    # self.trajectory_history[env_ids] = 0
                    # obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                    #                                     obs[:, self.env.privileged_dim + 9:-self.env.height_dim]),
                    #                                    dim=1)
                    # self.trajectory_history = torch.concat(
                    #     (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards  # 当前env_step完成后更新的 对应env的 所有奖励之和 (num_envs,)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

            # copy the config file
            # if it == 0:
            #     os.system("cp ./legged_gym/envs/a1/a1_amp_config.py " + self.log_dir + "/")

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        # self.writer.add_scalar('Loss/vel_predict', locs['mean_vel_predict_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_policy_pred', locs['mean_policy_pred'], locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_expert_pred', locs['mean_expert_pred'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          # f"""{'Vel predict loss:':>{pad}} {locs['mean_vel_predict_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {format_time(self.tot_time)}\n"""
                       f"""{'ETA:':>{pad}} {format_time((self.tot_time / (locs['it'] + 1 - self.current_learning_iteration)) * (self.current_learning_iteration + locs['num_learning_iterations'] - locs['it']))}\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            # 'world_model_dict': self._world_model.state_dict(),
            # 'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            # 'depth_predictor': self.depth_predictor.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])  # , strict=False
        # self._world_model.load_state_dict(loaded_dict['world_model_dict'], strict=False)
        # if(load_wm_optimizer):
        #     self._world_model._model_opt._opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
