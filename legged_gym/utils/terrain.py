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

import numpy as np
import random
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.new_terrains import add_extreme_gap_terrain

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots=None) -> None:

        self.cfg = cfg
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.int64)

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        # if cfg.curriculum:
        #     self.rough_terrain()
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
            difficulty = np.random.choice([0.1])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

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

    def rough_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0.00, 0.05)
            difficulty = np.random.choice([1.])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
            print("#############################################################")
    
    def add_roughness(self, terrain, difficulty=1.0):
        if not getattr(self.cfg, "add_roughness", True):
            return
        height_range = getattr(self.cfg, "height", [0.01, 0.010])
        downsampled_scale = getattr(self.cfg, "downsampled_scale", 0.2)
        max_height = (height_range[1] - height_range[0]) * difficulty + height_range[0]
        height = random.uniform(height_range[0], max_height)
        terrain_utils.random_uniform_terrain(
            terrain, min_height=-height, max_height=height,
            step=0.005, downsampled_scale=downsampled_scale)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        depth = 0.6
        terrain.idx = 0
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain.idx = 0
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain.idx = 1
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
                terrain.idx = 2
            else:
                terrain.idx = 3
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            terrain.idx = 4
        # ---- extreme_gap parkour (replaces random / gap / pit) ----
        elif choice < self.proportions[5]:
            # single-gap
            add_extreme_gap_terrain.parkour_step_gap_terrain(
                terrain, difficulty, depth=depth, platform_size=2.)
            self.add_roughness(terrain)
            terrain.idx = 5
        elif choice < self.proportions[6]:
            # step-stone
            stones_size = 0.7 if difficulty < 0.2 else -0.5 * difficulty * difficulty + 0.7
            stone_distance = 0.05 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            add_extreme_gap_terrain.stepping_stones_terrain(
                terrain, stone_size=stones_size, stone_distance=stone_distance,
                max_height=step_height, platform_size=2.0, depth=depth)
            self.add_roughness(terrain)
            terrain.idx = 6
        elif choice < self.proportions[7]:
            # Stones-2Rows
            stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
            stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            add_extreme_gap_terrain.stepping_two_stones_terrain(
                terrain, stone_size=stones_size, stone_distance=stone_distance,
                max_height=step_height, platform_size=2.0, depth=depth)
            self.add_roughness(terrain)
            terrain.idx = 7
        elif choice < self.proportions[8]:
            # stones-1Rows
            stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
            stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            add_extreme_gap_terrain.stepping_one_stones_terrain(
                terrain, difficulty=difficulty, stone_size=stones_size,
                stone_distance=stone_distance, max_height=step_height,
                platform_size=2.0, depth=depth)
            self.add_roughness(terrain)
            terrain.idx = 8
        elif choice < self.proportions[9]:
            # single-bridge
            stones_size = 0.6 if difficulty < 0.2 else -0.6 * difficulty + 0.8
            add_extreme_gap_terrain.stepping_one_bridge_terrain(
                terrain, stone_size=stones_size, difficulty=difficulty,
                platform_size=2.0, depth=depth)
            self.add_roughness(terrain)
            terrain.idx = 9
        elif choice < self.proportions[10]:
            # hurdle
            hurdel_height_mix = step_height if difficulty < 0.1 else 0.1 + 0.40 * difficulty
            hurdel_height_max = 0.1 + step_height if difficulty < 0.1 else 0.2 + 0.50 * difficulty
            stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
            stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
            add_extreme_gap_terrain.parkour_hurdle_terrain(
                terrain, difficulty=difficulty, stone_size=stones_size,
                stone_distance=stone_distance, max_height=step_height,
                hurdle_height_range=[hurdel_height_mix, hurdel_height_max],
                platform_size=2.0, depth=depth)
            self.add_roughness(terrain)
            terrain.idx = 10
        elif choice < self.proportions[11]:
            # ramp：难度线性插值，最高 33°（tanθ = strength * vs / hs）
            min_strength = 2.0
            max_strength = np.tan(np.deg2rad(33.0)) * self.cfg.horizontal_scale / self.cfg.vertical_scale
            slope_strength = min_strength + (max_strength - min_strength) * difficulty
            add_extreme_gap_terrain.half_sloped_terrain(
                terrain, level_index=difficulty * 2.5, depth=depth, slope_strength=slope_strength)
            self.add_roughness(terrain)
            terrain.idx = 11
        elif len(self.proportions) > 12 and choice < self.proportions[12]:
            # 上下楼梯（台阶宽 0.25m）：前上后下，高度同原始楼梯
            add_extreme_gap_terrain.half_stairs_terrain(
                terrain, step_width=0.25, step_height=step_height,
                platform_size=2.0, depth=depth)
            self.add_roughness(terrain)
            terrain.idx = 12
        elif len(self.proportions) > 13 and choice < self.proportions[13]:
            # 上下楼梯（台阶宽 0.30m）
            add_extreme_gap_terrain.half_stairs_terrain(
                terrain, step_width=0.30, step_height=step_height,
                platform_size=2.0, depth=depth)
            self.add_roughness(terrain)
            terrain.idx = 13
        elif len(self.proportions) > 14 and choice < self.proportions[14]:
            # 最后一列：纯粗糙地 ±rough_height（默认 ±3cm）
            h = getattr(self.cfg, "rough_height", 0.03)
            terrain_utils.random_uniform_terrain(
                terrain, min_height=-h, max_height=h, step=0.005, downsampled_scale=0.2)
            terrain.idx = 14
        else:
            # corridor
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.03, downsampled_scale=0.2)
            terrain.idx = 15

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

        idx = int(getattr(terrain, "idx", 0))
        self.terrain_type[i, j] = idx
        # 跑酷类（gap/石阵/桥/栏/坡/上下楼梯）：出生点放在入口平台中央，而不是块中心（中心常是坑）
        # 对齐 MGDP gap_parkour：入口约在前进方向 1m 处、横向居中
        parkour_ids = set(range(5, 14))  # 5~13
        if idx in parkour_ids:
            spawn_x_local = 1.0
            env_origin_x = i * self.env_length + spawn_x_local
            env_origin_y = (j + 0.5) * self.env_width
            x1 = max(0, int((spawn_x_local - 0.5) / terrain.horizontal_scale))
            x2 = min(terrain.width, int((spawn_x_local + 0.5) / terrain.horizontal_scale))
            y1 = max(0, int((self.env_width / 2. - 0.5) / terrain.horizontal_scale))
            y2 = min(terrain.length, int((self.env_width / 2. + 0.5) / terrain.horizontal_scale))
        else:
            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
            x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
            y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
            y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        # 入口若落在坑底（极端情况），抬到 0，避免出生点埋进地下
        if idx in parkour_ids:
            env_origin_z = max(float(env_origin_z), 0.0)
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

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
