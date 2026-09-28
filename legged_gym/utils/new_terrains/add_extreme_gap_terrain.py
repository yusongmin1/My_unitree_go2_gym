import numpy as np
import random

from isaacgym import terrain_utils


def trimesh_terrain(terrain, choice, difficulty, slope,
                    proportions, step_height, discrete_obstacles_height, stepping_stones_size,
                    stone_distance, gap_size, pit_depth, add_roughness, num_rows):
    depth = random.uniform(0.6, 0.6)
    if choice < proportions[0]:
        idx = 0
        add_roughness(terrain)
    elif choice < proportions[2]:
        if choice < proportions[1]:
            step_height *= -1
            idx = 1
        else:
            idx = 2
        pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=2.)

        add_roughness(terrain)

    elif choice < proportions[3]:
        idx = 3

        parkour_step_gap_terrain(terrain, difficulty, depth=depth, platform_size=2)
        add_roughness(terrain)
    elif choice < proportions[4]:
        idx = 4
        stones_size = 0.7 if difficulty < 0.2 else -0.5 * difficulty * difficulty + 0.7
        stone_distance = 0.05 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        stepping_stones_terrain(
            terrain,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth,
        )
        add_roughness(terrain)
    elif choice < proportions[5]:
        idx = 5

        # stones_size = 0.45 if difficulty < 0.2 else -0.3 * difficulty + 0.5
        # stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10

        stepping_two_stones_terrain(
            terrain,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)
        add_roughness(terrain)
    elif choice < proportions[6]:
        idx = 6
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.0 if difficulty < 0.2 else 0.2 * int(10 * difficulty) / 10

        stepping_two_discrete_stones_terrain(
            terrain,
            difficulty =difficulty,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)

        add_roughness(terrain)
    elif choice < proportions[7]:
        idx = 7
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10

        stepping_one_stones_terrain(
            terrain,
            difficulty=difficulty,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)

        add_roughness(terrain)
    elif choice < proportions[8]:
        idx = 8
        stones_size = 0.6 if difficulty < 0.2 else -0.6 * difficulty + 0.8
        stepping_one_bridge_terrain(
            terrain,
            stone_size=stones_size,
            difficulty=difficulty,
            platform_size=2.0,
            depth=depth)
        add_roughness(terrain)

    elif choice < proportions[9]:
        idx = 9
        bream_length = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        stepping_breams_terrain(
            terrain,
            difficulty=difficulty,
            stone_size=bream_length,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)
        add_roughness(terrain)
    elif choice < proportions[10]:
        idx = 10
        bream_length = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        stepping_breams_rot_terrain(
            terrain,
            difficulty= difficulty,
            stone_size=bream_length,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)
        add_roughness(terrain)

    elif choice < proportions[11]:
        idx = 11
        bream_length = 0.41 if difficulty < 0.1 else -0.32 * difficulty + 0.4
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10

        stepping_breams_naw_terrain(
            terrain,
            difficulty= difficulty,
            stone_size=bream_length,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)
        add_roughness(terrain)

    elif choice < proportions[12]:
        idx = 12
        bream_length = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
        stone_distance = 0.1 if difficulty < 0.2 else 0.1+ 0.4 * int(10 * difficulty) / 10
        stepping_breams_cross_terrain(
            terrain,
            difficulty=difficulty,
            stone_size=bream_length,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)

    elif choice < proportions[13]:
        idx = 13
        bream_length = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10

        stepping_air_breams_terrain(
            terrain,
            stone_size=bream_length,
            stone_distance=stone_distance,
            max_height=step_height,
            platform_size=2.0,
            depth=depth)
        add_breams_terrain(terrain, num_rows)
        add_roughness(terrain)

    elif choice < proportions[14]:
        idx = 14
        air_stone(terrain,difficulty, depth=depth)
        add_stone_terrain(terrain, num_rows)

        # add_roughness(terrain)

    elif choice < proportions[15]:
        idx = 15
        hurdel_height_mix = step_height if difficulty < 0.1 else 0.1 + 0.40 * difficulty
        hurdel_height_max = 0.1 + step_height if difficulty < 0.1 else 0.2 + 0.50 * difficulty
        stones_size = 0.8 if difficulty < 0.2 else -0.5 * difficulty + 0.8
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10

        parkour_hurdle_terrain(terrain,
            difficulty=difficulty,
            stone_size=stones_size,
            stone_distance=stone_distance,
            max_height=step_height,
            hurdle_height_range=[hurdel_height_mix,
                                hurdel_height_max],
            platform_size=2.0,
            depth=depth)
        add_roughness(terrain)


    elif choice < proportions[16]:
        idx = 16
        half_sloped_terrain(terrain, level_index=difficulty*2.5, depth= depth)
        add_roughness(terrain)
    elif choice < proportions[17]:
        idx = 17
        narrow_corridor(terrain,difficulty, depth=depth)
        add_narrow_terrain(terrain, num_rows)
        # add_roughness(terrain)

    # else:
        # idx = 20
        # step_height *= +1
        # pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=2)
        # add_roughness(terrain)
    terrain.idx = idx



def air_stone(terrain, difficulty, depth=2.0):  
    start_y, end_y = 20, terrain.length - 20  
    terrain.height_field_raw[:] = int(-depth / terrain.vertical_scale)
    terrain.height_field_raw[:, start_y:end_y] = 0


def get_muti_stone_trimeshes(center_position, num_rows, num_cols,
                             terrain_length, terrain_width, horizontal_scale, vertical_scale, border_size,
                             center_rpy=np.zeros(3),
                             box_size=np.zeros(3),
                             platform_size=2):
    frame_vertices, frame_triangles, = [], []

    terrain_length = terrain_length / horizontal_scale
    terrain_width = terrain_width / horizontal_scale

    max_height = 0.55  
    min_height = 0.18  

    for i in range(num_rows):
        difficulty = i / (num_rows - 1) if num_rows > 1 else 0

        height = max_height - (max_height - min_height) * difficulty

        box_size[0] = 4.5
        box_size[1] = np.random.randint(18, 25) / 10
        box_size[2] = np.random.randint(1, 5) / 10

        center_x = (terrain_length - box_size[0]) // 20
        center_y = (terrain_width - box_size[1] + terrain_width) // 20

        center = center_position[i, 0:3] + np.array([center_x, center_y, box_size[2]])


        pos_x = np.random.randint(1, 2) / 10

        center += np.array([pos_x, 0, height])
        center = np.round(center, 3)

        vertices_cur, triangles_cur = stone_trimesh(box_size, center, center_rpy)
        frame_vertices.append(vertices_cur)
        frame_triangles.append(triangles_cur)
        center -= np.array([pos_x, 0, height])

    return frame_vertices, frame_triangles



def stone_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
        rpy= np.zeros(3), # euler angle (in rad) not implemented yet.
    ):
    if not (rpy == 0).all():
        raise NotImplementedError("Only axis-aligned box triangle mesh is implemented")

    vertices = np.empty((8, 3), dtype= np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2
    vertices = np.round(vertices, 3)
    # print('ver', center_position,size, vertices[[1, 5, 3, 7], :], vertices[[0, 1, 4, 5], :] )
    # print("***** V ", vertices.shape)

    triangles = -np.ones((12, 3), dtype= np.uint32)
    triangles[0] = [0, 2, 1] #
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2] #
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6] #
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5] #
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4] #
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3] #
    triangles[11]= [3, 6, 7]
    # print("***** T ", triangles.shape)

    return vertices, triangles

def add_stone_terrain(terrain, num_rows):
    terrain.center_position_stone = np.zeros((num_rows, 3))

def add_narrow_terrain(terrain, num_rows):
    terrain.center_position = np.zeros((num_rows, 3))

def narrow_corridor(terrain, difficulty, depth=2.0): 
    
    platform_size = int(2 / terrain.horizontal_scale) 

    
    base_height = int(-depth / terrain.vertical_scale)  
    wall_abs_height = np.random.randint(200, 251)  

    
    start_y, end_y = 20, terrain.length - 20  
    terrain.height_field_raw[:] = base_height
    terrain.height_field_raw[:, start_y:end_y] = 0

    
    center_y = terrain.length // 2  

    
    min_gap = 2  
    max_gap = 8  

    
    narrow_gap_size = max_gap - (max_gap - min_gap) * difficulty
    
    narrow_gap = int(narrow_gap_size)  
    narrow_gap = np.clip(narrow_gap, min_gap, max_gap)  

    
    terrain.height_field_raw[platform_size + 20:, center_y + narrow_gap:end_y] = wall_abs_height
    terrain.height_field_raw[platform_size + 20:, start_y:center_y - narrow_gap] = wall_abs_height

def half_sloped_terrain(terrain, level_index, platform_size=2.,
                        slope_strength=None, final_platform_length=5, depth=2.0):
    
    platform_size = int(platform_size / terrain.horizontal_scale)

    
    if slope_strength is None:
        slope_strength = 2.0 + 2.5 * level_index  

    base_height = int(-depth / terrain.vertical_scale)

    
    start_y, end_y = 20, terrain.length - 20
    mid_region = (slice(None), slice(start_y, end_y))  

    
    terrain.height_field_raw[:] = base_height  
    terrain.height_field_raw[mid_region] = 0  

    
    positions = {
        'initial_platform': (0, platform_size),
        'up_slope': (platform_size, 2 * platform_size),
        'mid_platform': (2 * platform_size, 3 * platform_size),
        'down_slope': (3 * platform_size, 4 * platform_size),
        'final_platform': (4 * platform_size, terrain.width)
    }

    
    positions['final_platform'] = (
        positions['down_slope'][1],
        min(positions['down_slope'][1] + final_platform_length, terrain.width)
    )

    
    up_start, up_end = positions['up_slope']
    xs = np.arange(up_start, up_end)
    
    max_height = slope_strength * (up_end - up_start)
    up_heights = (slope_strength * (xs - up_start)).astype(np.int16)
    terrain.height_field_raw[up_start:up_end, start_y:end_y] = up_heights[:, None]
    
    mid_start, mid_end = positions['mid_platform']
    terrain.height_field_raw[mid_start:mid_end, start_y:end_y] = max_height


    
    down_start, down_end = positions['down_slope']
    down_end = min(down_start + (up_end - up_start), positions['final_platform'][0])
    xs = np.arange(down_start, down_end)
    down_heights = (max_height - slope_strength * (xs - down_start)).astype(np.int16)
    terrain.height_field_raw[down_start:down_end, start_y:end_y] = down_heights[:, None]


def half_stairs_terrain(terrain, step_width, step_height, platform_size=2.,
                        final_platform_length=5, depth=0.6):
    """类似 ramp 的纵向布局：入口平台 → 上楼梯 → 中平台 → 下楼梯 → 尾平台。
    step_width / step_height 单位为米；高度沿前进方向（terrain.width）累加。
    """
    step_w = max(1, int(round(step_width / terrain.horizontal_scale)))
    step_h = max(1, int(round(step_height / terrain.vertical_scale)))
    platform = int(platform_size / terrain.horizontal_scale)
    base_height = int(-depth / terrain.vertical_scale)

    start_y, end_y = 20, terrain.length - 20
    terrain.height_field_raw[:] = base_height
    terrain.height_field_raw[:, start_y:end_y] = 0

    # 入口平台
    terrain.height_field_raw[0:platform, start_y:end_y] = 0

    # 上楼梯：platform → 2*platform
    up_start, up_end = platform, 2 * platform
    height = 0
    x = up_start
    while x < up_end:
        height += step_h
        x_next = min(x + step_w, up_end)
        terrain.height_field_raw[x:x_next, start_y:end_y] = height
        x = x_next
    max_height = height

    # 中平台
    mid_start, mid_end = 2 * platform, 3 * platform
    terrain.height_field_raw[mid_start:mid_end, start_y:end_y] = max_height

    # 下楼梯：3*platform 起，与上坡同长
    down_start = 3 * platform
    down_end = min(down_start + (up_end - up_start), terrain.width)
    height = max_height
    x = down_start
    while x < down_end and height > 0:
        x_next = min(x + step_w, down_end)
        terrain.height_field_raw[x:x_next, start_y:end_y] = height
        height = max(0, height - step_h)
        x = x_next

    # 尾平台归零
    final_start = min(max(x, 4 * platform), terrain.width)
    final_end = min(final_start + int(final_platform_length / terrain.horizontal_scale), terrain.width)
    terrain.height_field_raw[final_start:final_end, start_y:end_y] = 0
    if final_end < terrain.width:
        terrain.height_field_raw[final_end:terrain.width, start_y:end_y] = 0

    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):

    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = platform_size + 20
    stop_x = int(terrain.width) - 20
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain

def box_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
        rpy= np.zeros(3), # euler angle (in rad) not implemented yet.
    ):
    if not (rpy == 0).all():
        raise NotImplementedError("Only axis-aligned box triangle mesh is implemented")

    vertices = np.empty((8, 3), dtype= np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2
    vertices = np.round(vertices, 3)
    # print('ver', center_position,size, vertices[[1, 5, 3, 7], :], vertices[[0, 1, 4, 5], :] )
    # print("***** V ", vertices.shape)

    triangles = -np.ones((12, 3), dtype= np.uint32)
    triangles[0] = [0, 2, 1] #
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2] #
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6] #
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5] #
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4] #
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3] #
    triangles[11]= [3, 6, 7]
    # print("***** T ", triangles.shape)

    return vertices, triangles

def parkour_step_gap_terrain(terrain, difficulty, depth,  platform_size=2.):
    gap_size = int(difficulty / terrain.horizontal_scale)
    # gap_size = np.clip(int(gap_size/2), 2, 40)

    gap_size = np.clip(int(gap_size), 2, 12)

    depth = int(depth / terrain.vertical_scale)

    platform_size = int(platform_size / terrain.horizontal_scale)
    start_y = 0
    end_y = int(terrain.length-platform_size/8)

    start_x = platform_size
    center_x = terrain.width // 2
    terrain.height_field_raw[start_x: center_x, start_y: end_y] = -depth
    terrain.height_field_raw[start_x+gap_size: center_x - gap_size, start_y+gap_size: end_y-gap_size] = 0



    start_x = center_x+int(platform_size/2)
    end_x = int(terrain.width)
    terrain.height_field_raw[start_x: end_x, start_y: end_y] = -depth
    terrain.height_field_raw[start_x+gap_size: end_x - gap_size, start_y+gap_size: end_y-gap_size] = 0

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=1):

    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance_x = int(stone_distance / terrain.horizontal_scale)
    stone_distance_y = min(int(stone_distance / terrain.horizontal_scale), 1)
    max_height = min(int(max_height / terrain.vertical_scale), 40)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(1, max_height, step=4)

    start_x = 0
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)


    if terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance_y)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length - 10:
                stop_y = min(terrain.length - 10, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance_y
            start_x += stone_size + stone_distance_x

    x1 = 0
    x2 = platform_size
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def stepping_two_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=1):
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)

    max_height = int(max_height / terrain.vertical_scale)
    max_height = np.clip(max_height, 0, 30)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height_range = np.arange(1, max_height, step=3)

    stone_size = np.clip(stone_size, 6, 16)

    stone_distance_gap = int(stone_distance / terrain.horizontal_scale)
    stone_distance_gap = np.clip(stone_distance_gap, 1, 6)
    stone_distance_range = np.arange(0, stone_distance_gap, step=1) + 1

    # print('sfsf', stone_size, stone_distance_range, height_range)

    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2
    row1_y = int(platform_y + platform_size/2 - stone_size - 0.1)
    row2_y = int(platform_y + platform_size/2 + 0.1)

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    # Generate the series of stones in front of the platform
    start_x = 0
    while start_x < terrain.width:
        stop_x = min(terrain.width, start_x + stone_size)
        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + stone_size] = np.random.choice(height_range)
        terrain.height_field_raw[start_x: stop_x, row2_y: row2_y + stone_size] = np.random.choice(height_range)
        stone_distance_gap = np.random.choice(stone_distance_range)
        start_x += stone_size + stone_distance_gap

    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_two_discrete_stones_terrain(terrain, difficulty, stone_size, stone_distance, max_height, platform_size=1., depth=1):
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    # stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    max_height = np.clip(max_height, 0, 30)

    platform_size = int(platform_size / terrain.horizontal_scale)

    height_range = np.arange(0, max_height, step=3)
    stone_size = np.clip(stone_size, 6, 16)

    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2
    row1_y = int(platform_y + platform_size / 2 - stone_size - 0.01)
    row2_y = int(platform_y + platform_size / 2 + 0.01)

    if difficulty < 0.4:
        random_stone_gap = random.randint(1, 2)
        stone_distance = int(stone_size / 2) + random_stone_gap
    else:
        random_stone_gap = random.randint(1, 3)
        stone_distance = int(stone_size / 2) + random_stone_gap
    # print('step, ', stone_distance)
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    # Generate the series of stones in front of the platform
    start_x = 0
    is_left = np.random.choice([True, False])
    while start_x < terrain.width:
        stop_x = min(terrain.width, start_x + stone_size)
        height = np.random.choice(height_range)
        if is_left:
            for x in range(start_x, stop_x):
                for y in range(row1_y, row1_y + stone_size):
                    terrain.height_field_raw[x, y] = height
        else:
            for x in range(start_x, stop_x):
                for y in range(row2_y, row2_y + stone_size):
                    terrain.height_field_raw[x, y] = height
        start_x += stone_distance
        is_left = not is_left

    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_one_stones_terrain(terrain, difficulty, stone_size, stone_distance, max_height, platform_size=1., depth=1):
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    max_height = np.clip(max_height, 0, 20)

    platform_size = int(platform_size / terrain.horizontal_scale)

    if difficulty < 0.3:
        height_range = np.arange(1, max_height, step=3)
    elif 0.3 <= difficulty < 0.7:
        height_range = np.arange(1, max_height, step=4)
    else:
        height_range = np.arange(1, max_height, step=12)

    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    stone_distance_gap = np.clip(int(difficulty / terrain.horizontal_scale), 4,
                                 16)

    # print('stone',difficulty, stone_distance_gap)
    # print()
    # Generate the series of stones in front of the platform
    start_x = 0
    while start_x < terrain.width:
        stone_size_x = np.clip(stone_size, 12, 30)
        stone_distance_range = np.arange(12, 30, step=1)
        stone_size_y = np.random.choice(stone_distance_range)

        row1_y = int(platform_y + platform_size / 2 - stone_size_y / 2)



        stop_x = min(terrain.width, start_x + stone_size_x)
        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + stone_size_y] = np.random.choice(height_range)
        start_x += stone_size + stone_distance_gap

    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_one_bridge_terrain(terrain, stone_size,  difficulty, platform_size=1., depth=1):

    stone_size = int(stone_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2
    row1_y = int(platform_y + platform_size/2 - stone_size/2)

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)

    height_range = np.arange(0, 10, step=2)

    # print('stone_size', stone_size)
    # print()
    # Generate the series of stones in front of the platform
    start_x = 0
    while start_x < terrain.width:
        stop_x = min(terrain.width, start_x + stone_size)
        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + stone_size] = np.random.choice(height_range)
        start_x += stone_size

    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_breams_terrain(terrain, difficulty, stone_size,  stone_distance, max_height, platform_size=2., depth=1):
    # switch parameters to discrete units

    bream_length = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    if difficulty < 0.3:
        height_range = np.arange(1, max_height, step=4)
    elif 0.3 <= difficulty < 0.7:
        height_range = np.arange(1, max_height, step=8)
    else:
        height_range = np.arange(1, max_height, step=12)

    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    min_bream_width = 15
    max_bream_width = 30

    tilt_direction = random.choice([-1, 1])
    if tilt_direction==1:
        rotation_angle = 1 / 2
    else:
        rotation_angle = 1 / 4
    rotation_breams = False  # np.random.choice([True, False])

    
    start_x = 0
    while start_x < terrain.width:
        bream_width = random.randint(min_bream_width, max_bream_width)
        row1_y = int(platform_y + platform_size / 2 - bream_width / 2)
        stop_x = min(terrain.width, start_x + bream_length)
        height = np.random.choice(height_range)
        if rotation_breams:
            
            for y in range(row1_y, row1_y + bream_width):
                x_offset = int(rotation_angle * (y - row1_y) * tilt_direction)
                terrain.height_field_raw[start_x + x_offset : stop_x + x_offset , y] = height
        else:
            terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + bream_width] = height

        start_x += bream_length + stone_distance

    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_breams_rot_terrain(terrain, difficulty, stone_size,  stone_distance, max_height, platform_size=2., depth=1):
    # switch parameters to discrete units

    bream_length = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    if difficulty < 0.3:
        height_range = np.arange(1, max_height, step=4)
    elif 0.3 <= difficulty < 0.7:
        height_range = np.arange(1, max_height, step=8)
    else:
        height_range = np.arange(1, max_height, step=12)

    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    min_bream_width = 20
    max_bream_width = 50

    # max_height = 10
    if difficulty < 0.2:
        bream_length = random.choice([4, 3])
        height_range = np.arange(1, max_height, step=1)
    elif 0.2 <= difficulty < 0.8:
        bream_length = 3
        height_range = np.arange(1, max_height, step=1)
    else:
        bream_length = 2
        height_range = np.arange(1, max_height, step=1)

    if difficulty < 0.4:
        stone_distance = random.randint(4, 8)
    elif 0.4 <= difficulty < 0.8:
        stone_distance = random.randint(6, 9)
    elif 0.8 <= difficulty:
        stone_distance = random.randint(7, 12)

    tilt_direction = random.choice([-1, 1])
    if tilt_direction==1:
        rotation_angle = 1 / 2
    else:
        rotation_angle = 1 / 4

    if difficulty < 0.4:
        rotation_breams = False  # np.random.choice([True, False])
    else:
        rotation_breams = True  # np.random.choice([True, False])

    
    # start_x = 0
    # while start_x < terrain.width:
    #     bream_width = random.randint(min_bream_width, max_bream_width)
    #     row1_y = int(platform_y + platform_size / 2 - bream_width / 2)
    #     stop_x = min(terrain.width, start_x + bream_length)
    #     height = np.random.choice(height_range)
    #     if rotation_breams:
    
    #         for y in range(row1_y, row1_y + bream_width):
    #             x_offset = int(rotation_angle * (y - row1_y) * tilt_direction)
    #             terrain.height_field_raw[start_x + x_offset : stop_x + x_offset , y] = height
    #     else:
    #         terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + bream_width] = height
    #
    #     start_x += bream_length + stone_distance

    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_breams_naw_terrain(terrain, difficulty, stone_size,  stone_distance, max_height, platform_size=2., depth=1):
    bream_length = int(stone_size / terrain.horizontal_scale)
    # stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    bream_length = np.clip(bream_length, 0, 2)

    max_height = 10
    if difficulty < 0.5:
        bream_length = random.choice([3, 3])
        height_range = np.arange(1, max_height, step=1)
    elif 0.5 <= difficulty < 0.8:
        bream_length = 3
        height_range = np.arange(1, max_height, step=1)
    else:
        bream_length = 2
        height_range = np.arange(1, max_height, step=1)
    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2

    # bream_length = 3
    # print('bream_length',max_height, bream_length, stone_distance)

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    min_bream_width = 20
    max_bream_width = 60

    
    start_x = 0
    while start_x < terrain.width:
        if difficulty < 0.4:
            stone_distance = random.randint(6, 8)
        elif 0.4 <= difficulty < 0.8:
            stone_distance = random.randint(7, 12)
        elif 0.8 <= difficulty:
            stone_distance = random.randint(7, 12)

        bream_width = random.randint(min_bream_width, max_bream_width)
        row1_y = int(platform_y + platform_size / 2 - bream_width / 2)
        stop_x = min(terrain.width, start_x + bream_length)
        height = np.random.choice(height_range)
        # print('gap', stone_distance, bream_length)

        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + bream_width] = height

        start_x += bream_length + stone_distance

    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_breams_cross_terrain(terrain, difficulty, stone_size,  stone_distance, max_height, platform_size=2., depth=1):
    # switch parameters to discrete units


    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height_range = np.arange(0, max_height, step=4)
    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2

    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    min_bream_width = 30
    max_bream_width = 60

    x_stone_distance =  random.randint(1, 3)
    if difficulty < 0.3:
        y_stone_distance =  random.randint(4, 7)
        stone_distance = random.randint(2, 4)
        bream_length =  random.randint(6, 8)
    elif 0.3<=difficulty < 0.5:
        y_stone_distance =  random.randint(3, 5)
        stone_distance = random.randint(3, 5)
        bream_length =  random.randint(3, 5)
    elif 0.5<=difficulty < 0.8:
        x_stone_distance = random.randint(3, 5)
        y_stone_distance =  random.randint(3, 5)
        stone_distance = random.randint(4, 6)
        bream_length =  random.randint(3, 5)
    else:
        y_stone_distance = random.randint(2, 2)
        stone_distance = random.randint(4, 5)
        bream_length = random.randint(2, 2)


    
    start_x = platform_size
    while start_x < terrain.width:
        bream_width = random.randint(min_bream_width, max_bream_width)
        row1_y = int(platform_y + platform_size / 2 - bream_width / 2)
        stop_x = min(terrain.width, start_x + bream_length)

        terrain.height_field_raw[start_x: stop_x, row1_y: row1_y + bream_width] = 0
        start_x += bream_length + stone_distance



    center_y = int(platform_y + platform_size / 2)

    start_y1 = center_y - x_stone_distance
    start_y2 = center_y + x_stone_distance

    terrain.height_field_raw[platform_size: terrain.width, start_y1- y_stone_distance: start_y1] = 0
    terrain.height_field_raw[platform_size: terrain.width, start_y2: start_y2 + y_stone_distance] = 0


    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0

    return terrain

def stepping_air_breams_terrain(terrain, stone_size,  stone_distance, max_height, platform_size=1., depth=1):
    # switch parameters to discrete units

    platform_size = int(platform_size / terrain.horizontal_scale)
    # Calculate the y-coordinates for the two rows of stones
    platform_y = (terrain.length - platform_size) // 2
    terrain.height_field_raw[:, :] = int(-depth / terrain.vertical_scale)
    # Create the platform
    x1 = 0
    x2 = platform_size
    terrain.height_field_raw[x1:x2, platform_y:platform_y + platform_size] = 0
    return terrain

def get_muti_beam_trimeshes(center_position,  num_rows, num_cols,
                      terrain_length, terrain_width, horizontal_scale, vertical_scale, border_size,
                      center_rpy=np.zeros(3),
                      box_size=np.zeros(3),
                      platform_size=2):  # goals: [rows, cols, goals, 3]
    # create mulitple frame trimesh
    frame_vertices, frame_triangles, = [], []
    platform_size = int(platform_size / horizontal_scale)
    box_size[2] = 0.1
    min_bream_width = 7;  max_bream_width = 15


    terrain_length = terrain_length / horizontal_scale
    terrain_width = terrain_width / horizontal_scale
    for i in range(num_rows):
        difficulty = i / (num_rows-1)
        stone_size = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
        stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
        bream_length = int(stone_size / horizontal_scale)
        stone_distance = int(stone_distance / horizontal_scale)
        box_size[0] = bream_length/20

        step_height = 0.04 if difficulty < 0.2 else 0.05 + 0.18 * difficulty
        max_height = int(step_height / vertical_scale)/100

        height_range = np.arange(0, max_height, step=0.04)

        center = center_position[i, 0:3] + np.array([5+box_size[0]/2, 5, 0])
        start_y = (terrain_width - platform_size) // 2
        start_x = platform_size
        while start_x < terrain_length:
            # bream_width = random.randint(min_bream_width, max_bream_width)
            bream_width = 2 * random.randint(min_bream_width, max_bream_width)

            box_size[1] = bream_width/20
            # height = random.randint(-max_height, max_height)/200
            height = np.random.choice(height_range)

            center += np.array([start_x/20, start_y/10, height])
            center = np.round(center, 3)
            vertices_cur, triangles_cur = box_trimesh(box_size, center, center_rpy)
            frame_vertices.append(vertices_cur)
            frame_triangles.append(triangles_cur)
            center -= np.array([start_x/20, start_y/10, height])
            start_x += bream_length + stone_distance


    return frame_vertices, frame_triangles

def add_breams_terrain(terrain, num_rows):
    goals = np.zeros((num_rows, 3))
    terrain.center_position = goals


# def get_box_trimeshes(center_position,  num_rows, num_cols,
#                       terrain_length, terrain_width, horizontal_scale, vertical_scale, border_size,
#                       center_rpy=np.zeros(3),
#                       box_size=np.zeros(3),
#                       platform_size=2):  # goals: [rows, cols, goals, 3]
#     # create mulitple frame trimesh
#     frame_vertices, frame_triangles, = [], []
#     # [5, 24+5+2, 0]
#     center_position += np.array([0, 6*4+box_size[1], 0])
#     box_size[2] = 0.1
#     # max_height = int(max_height / vertical_scale)
#     min_bream_width = 15;  max_bream_width = 30
#     terrain_length = terrain_length / horizontal_scale
#     terrain_width = terrain_width / horizontal_scale
#
#     for i in range(num_rows):
#         difficulty = i / (num_rows-1)
#         stone_size = 0.35 if difficulty < 0.2 else -0.1 * difficulty + 0.35
#         stone_distance = 0.1 if difficulty < 0.2 else 0.4 * int(10 * difficulty) / 10
#         bream_length = int(stone_size / horizontal_scale)
#         stone_distance = int(stone_distance / horizontal_scale)
#         platform_size = int(platform_size / horizontal_scale)
#         box_size[0] = bream_length/20
#         start_y = (terrain_width - platform_size) // 2
#
#         start_x = (terrain_length + platform_size) // 2
#         while start_x < terrain_length:
#             bream_width = random.randint(min_bream_width, max_bream_width)
#
#             box_size[1] = bream_width/20
#
#             center_position += np.array([start_x/20, start_y/10, 0])
#
#             vertices_cur, triangles_cur = box_trimesh(box_size, center_position, center_rpy)
#             frame_vertices.append(vertices_cur)
#             frame_triangles.append(triangles_cur)
#             center_position -= np.array([start_x/20, start_y/10, 0])
#             start_x += bream_length + stone_distance
#
#         start_x = 0
#         while start_x < (terrain_length - platform_size) // 2:
#             bream_width = random.randint(min_bream_width, max_bream_width)
#             box_size[1] = bream_width / 20
#             center_position += np.array([start_x / 20, start_y / 10, 0])
#             vertices_cur, triangles_cur = box_trimesh(box_size, center_position, center_rpy)
#             frame_vertices.append(vertices_cur)
#             frame_triangles.append(triangles_cur)
#             center_position -= np.array([start_x / 20, start_y / 10, 0])
#
#             start_x += bream_length + stone_distance
#
#     return frame_vertices, frame_triangles

def parkour_step_terrain(terrain, num_stones=8, x_range=[0.2, 0.4], hurdle_height_range=[0.1, 0.2], platform_size=1.5):

    
    platform_size = int(platform_size / terrain.horizontal_scale)
    dis_x_min = round((x_range[0]) / terrain.horizontal_scale)
    dis_x_max = round((x_range[1]) / terrain.horizontal_scale)

    # step_height = round(step_height / terrain.vertical_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)
    step_height = np.random.randint(hurdle_height_min, hurdle_height_max)
    # print('sf',step_height, hurdle_height_min, hurdle_height_max)
    
    max_x = int((terrain.width + platform_size) )

    
    start_y = 2
    end_y = int(terrain.length) - 2

    new_stair_height = 0

    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)

        start_x = np.random.randint(60, 65)

        new_stair_height += step_height

        end_x = start_x + rand_x
        end_x = np.clip(end_x, start_x, max_x)
        terrain.height_field_raw[start_x:end_x, start_y:end_y] = new_stair_height

    return terrain

def parkour_hurdle_terrain(terrain, difficulty, stone_size, stone_distance, max_height, hurdle_height_range, platform_size=1., depth=1):
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    pit = int(-depth / terrain.vertical_scale)

    # 走廊与跨栏均拉满地形横向宽度（terrain.length ↔ terrain_width）
    y0, y1 = 0, terrain.length

    terrain.height_field_raw[:, :] = pit
    terrain.height_field_raw[:, y0:y1] = 0

    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)
    # randint 上界开区间；保证 min < max
    if hurdle_height_max <= hurdle_height_min:
        hurdle_height_max = hurdle_height_min + 1
    step_height = np.random.randint(hurdle_height_min, hurdle_height_max)

    # 栏杆沿前进方向排布；横向宽度 = 整幅地形宽
    start_x = platform_size + 20
    while start_x < terrain.width - 20:
        stone_size_x = np.clip(stone_size, 12, 30)
        stop_x = min(terrain.width, start_x + stone_size_x)
        terrain.height_field_raw[start_x:stop_x, y0:y1] = step_height

        stone_distance_gap = np.clip(int(difficulty / terrain.horizontal_scale), 60, 80)
        start_x += stone_size + stone_distance_gap

    return terrain
