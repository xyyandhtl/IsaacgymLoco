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
import torch
import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length  # 8m
        self.env_width = cfg.terrain_width  # 8m
        self.xSize = cfg.terrain_length * cfg.num_rows  # 地形总长度 = 8 * 10
        self.ySize = cfg.terrain_width * cfg.num_cols  # 地形总宽度 = 8 * 10
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols  # 子地形的个数，10 * 20
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)  # 每个子地形在宽度方向上的 网格数量，8 / 0.1 = 80
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)  # 每个子地形在长度方向上的 网格数量，8 / 0.1 = 80

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)  # 边界所占的网格数量，15 / 0.1 = 150
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border  # 总地形网格（水平方向）的列数 20 * 80 + 2 * 150
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border  # 总地形网格的行数 10 * 80 + 2 * 150

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.7, 0.8])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):  # [0, 20]
            for i in range(self.cfg.num_rows):  # [0, 10]
                difficulty = i / self.cfg.num_rows  # [0.0, 0.1, ..., 0.9]
                choice = j / self.cfg.num_cols + 0.001  # [0.0+0.001, 0.05+0.001, ..., 0.95+0.001]

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        """
            choice: [0.0+0.001, 0.05+0.001, ..., 0.95+0.001]
            difficulty: [0.0, 0.1, ..., 0.9]
        """
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,  # 80
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,  # 0.005
                                horizontal_scale=self.cfg.horizontal_scale)  # 0.1
        slope = min(difficulty * 0.5, 0.4)  # [0.0, ..., 0.40]
        amplitude = min(0.02 + 0.1 * difficulty, 0.06)  # [0.02, 0.03..., 0.06]

        def calculate_step_height(difficulty):
            # [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.16, 0.175, 0.19, 0.205]
            if difficulty < 0.5:
                step_height = 0.06 + 0.2 * difficulty
            elif difficulty in [0.5, 0.6]:
                step_height = 0.16
            else:
                height_at_06 = 0.16
                step_height = height_at_06 + 0.15 * (difficulty - 0.6)
            return step_height
        step_height = calculate_step_height(difficulty)

        discrete_obstacles_height = 0.06 + difficulty * 0.15  # [0.06, 0.075, ..., 0.135, 0.15, ..., 0.195]
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = min(0.5 * difficulty, 0.35)

        num_rectangles = 20
        rectangle_min_size = 1.
        rectangle_max_size = 2.
        #
        # length = int(terrain.height_field_raw.shape[1] * 0.3)
        # terrain.height_field_raw[:, :length] = 250
        # terrain.height_feld_raw[:, -length:] = 250
        #
        # if random.random() > min(0.8, 0.2 + 0.5*difficulty):
        #     obstacle_width = int(terrain.height_field_raw.shape[1] * (0.3 + random.random() * 0.15))
        #     obstacle_length = int(terrain.height_field_raw.shape[0] * random.random() * 0.5)
        #     start = int(terrain.height_field_raw.shape[0] * random.random() * 0.6)
        #     if random.random() > 0.5:
        #         terrain.height_field_raw[start:start + obstacle_length, :obstacle_width] = 250
        #     else:
        #         terrain.height_field_raw[start:start + obstacle_length, -obstacle_width:] = 250
        #
        #
        # for i in range(2):
        #     pit_width = random.choice([2,3,4,5])
        #     start_x = min(int(terrain.height_field_raw.shape[0] * random.random() * 0.6), terrain.height_field_raw.shape[0] - 5)
        #     start_y = int(terrain.height_field_raw.shape[1] * 0.3) + \
        #               min(int(terrain.height_field_raw.shape[1] * random.random() * 0.4),terrain.height_field_raw.shape[1] - 5)
        #
        #     if random.random() > 0.5:
        #         terrain.height_field_raw[start_x:start_x + pit_width, start_y:start_y + pit_width] = 3 + 2 * random.random()
        #     else:
        #         terrain.height_field_raw[start_x:start_x + pit_width, start_y:start_y + pit_width] = -0.5 - 1 * random.random()
        if choice < self.proportions[0]:  # 平地
            flat_terrain(terrain)
        elif choice < self.proportions[1]:  # 粗糙地面
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[2]:  # 斜坡地形
            if choice < self.proportions[0] / 2:  # 斜坡地形中的前一半列，上斜坡
                slope *= -1
            # 后一半列，下斜坡
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[3]:  # 斜坡+粗糙地面起伏地形
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[5]:  # 台阶地形
            if choice<self.proportions[4]:  # 上台阶
                step_height *= -1
            # 下台阶
            pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3., border_width=1.)
        elif choice < self.proportions[6]:  # 离散障碍物
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[7]:  # 跳跃石地形
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0.8, platform_size=4.)
        elif choice < self.proportions[8]:  # 坑
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        else:  # 间隙
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def in_terrain_range(self, pos, device="cpu"):
        """ Check if the given position still have terrain underneath. (same x/y, but z is different)
            pos: (batch_size, 3) torch.Tensor
        """
        return torch.logical_and(
            pos[..., :2] >= 0,
            pos[..., :2] < torch.tensor([self.xSize, self.ySize], device=device),
        ).all(dim=-1)

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth



def flat_terrain(terrain):
    terrain.height_field_raw = np.zeros((terrain.width, terrain.length))


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1., border_width=0.):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        border_width (float): 地形周围平地的宽度 [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    step_width = round(step_width / terrain.horizontal_scale)
    step_height = round(step_height / terrain.vertical_scale)
    platform_size = round(platform_size / terrain.horizontal_scale)
    border_width = round(border_width / terrain.horizontal_scale)

    height = 0
    start_x = border_width
    stop_x = terrain.width - border_width
    start_y = border_width
    stop_y = terrain.length - border_width
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height

    terrain.height_field_raw[0:border_width, :] = 0
    terrain.height_field_raw[-border_width:, :] = 0
    terrain.height_field_raw[:, 0:border_width] = 0
    terrain.height_field_raw[:, -border_width:] = 0

    return terrain