"""AMP DreamWaQ 无头 play：每种地形存 3 张图，每张是原始 Logger 3×3 的一行。

第 1 行：基座速度 x / y / 偏航
第 2 行：关节位置 / 关节速度 / 基座速度 z
第 3 行：垂直接触力 / 力矩-速度 / 力矩
横轴只留数值，不写“时间”。

    python legged_gym/scripts/play_amp_dreamwaq_log.py --task=go2_amp_dreamwaq --headless
"""
import os

import isaacgym  # noqa: F401
import matplotlib
matplotlib.use('Agg')

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch

TERRAIN_NAMES = {
    0: '平缓斜坡',
    1: '粗糙斜坡',
    2: '下楼梯',
    3: '上楼梯',
    4: '离散障碍',
}
TERRAIN_SLUGS = {
    0: 'slope_smooth',
    1: 'slope_rough',
    2: 'stairs_down',
    3: 'stairs_up',
    4: 'discrete',
}
FOOT_LABELS = ['左前足', '右前足', '左后足', '右后足']
JOINT_INDEX = 1  # 与 play_dreamwaq 的 Logger 一致，记录 FL 大腿


def _setup_font():
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for name in ['Noto Serif CJK SC', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'SimHei']:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [name]
            break
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18


def column_terrain_idx(num_cols, proportions):
    """与 Terrain.make_terrain 的 choice 分支一致。"""
    p = np.cumsum(np.asarray(proportions, dtype=np.float64))
    idxs = []
    for j in range(num_cols):
        choice = j / num_cols + 0.001
        if choice < p[0]:
            idx = 0
        elif len(p) > 1 and choice < p[1]:
            idx = 1
        elif len(p) > 3 and choice < p[3]:
            idx = 2 if choice < p[2] else 3
        elif len(p) > 4 and choice < p[4]:
            idx = 4
        else:
            idx = int(np.searchsorted(p, choice, side='left'))
        idxs.append(idx)
    return np.asarray(idxs, dtype=np.int64)


def _save_logger_row(path, panels):
    """一张图 = 原始 3×3 的一行，三个子图横排。横轴不写“时间”。"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for ax, (title, ylabel, xlabel, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(True, alpha=0.35)
        ax.legend(loc='upper right', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_logger_rows(out_dir, slug, t, log):
    """log 的键与 legged_gym.utils.logger.Logger._plot 的 3×3 一一对应。"""
    jv = log['dof_vel']
    rows = [
        ('row1', [
            ('基座速度 x', '线速度（米/秒）', '', lambda ax: (
                ax.plot(t, log['base_vel_x'], label='实测'),
                ax.plot(t, log['command_x'], linestyle='--', label='指令'))),
            ('基座速度 y', '线速度（米/秒）', '', lambda ax: (
                ax.plot(t, log['base_vel_y'], label='实测'),
                ax.plot(t, log['command_y'], linestyle='--', label='指令'))),
            ('基座偏航速度', '角速度（弧度/秒）', '', lambda ax: (
                ax.plot(t, log['base_vel_yaw'], label='实测'),
                ax.plot(t, log['command_yaw'], linestyle='--', label='指令'))),
        ]),
        ('row2', [
            ('关节位置', '角度（弧度）', '', lambda ax: (
                ax.plot(t, log['dof_pos'], label='实测'),
                ax.plot(t, log['dof_pos_target'], linestyle='--', label='目标'))),
            ('关节速度', '角速度（弧度/秒）', '', lambda ax: ax.plot(t, jv, label='实测')),
            ('基座速度 z', '线速度（米/秒）', '', lambda ax: ax.plot(t, log['base_vel_z'], label='实测')),
        ]),
        ('row3', [
            ('垂直接触力', '力（牛）', '', lambda ax: [
                ax.plot(t, col, label=lab) for col, lab in zip(log['contact_forces_z'].T, FOOT_LABELS)
            ]),
            ('力矩-速度', '力矩（牛米）', '关节速度（弧度/秒）', lambda ax: ax.plot(
                jv, log['dof_torque'], 'x', markersize=3, label='实测')),
            ('力矩', '力矩（牛米）', '', lambda ax: ax.plot(t, log['dof_torque'], label='实测')),
        ]),
    ]
    paths = []
    for name, panels in rows:
        path = os.path.join(out_dir, f'{slug}_{name}.png')
        _save_logger_row(path, panels)
        paths.append(path)
    return paths


def pin_one_robot_per_terrain(env):
    """课程生成地形后关掉课程，每种地形放一台机器人。"""
    col_idx = column_terrain_idx(env.cfg.terrain.num_cols, env.cfg.terrain.terrain_proportions)
    picked = {}
    for j, idx in enumerate(col_idx):
        if int(idx) not in picked and int(idx) in TERRAIN_NAMES:
            picked[int(idx)] = j
    order = sorted(picked)
    if len(order) < len(TERRAIN_NAMES):
        raise RuntimeError(f'地形列没有覆盖全部类别，只找到 {order}')
    cols = [picked[k] for k in order]
    level = min(4, env.cfg.terrain.num_rows - 1)
    n = len(cols)
    env.cfg.terrain.curriculum = False
    env.terrain_levels[:n] = level
    env.terrain_types[:n] = torch.tensor(cols, device=env.device, dtype=torch.long)
    env.env_origins[:n] = env.terrain_origins[env.terrain_levels[:n], env.terrain_types[:n]]
    env.reset()
    return order


def play(args, mesh='trimesh'):
    _setup_font()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    plane = mesh == 'plane'

    env_cfg.env.num_envs = 1 if plane else len(TERRAIN_NAMES)
    # 崎岖地形必须先按课程生成，列号才对应类别；平地直接用 plane
    env_cfg.terrain.mesh_type = 'plane' if plane else 'trimesh'
    env_cfg.terrain.curriculum = not plane
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.max_init_terrain_level = 4
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1e9
    if hasattr(env_cfg.env, 'reference_state_initialization'):
        env_cfg.env.reference_state_initialization = False

    args.headless = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    if plane:
        env.reset()
        slugs = ['plane']
    else:
        order = pin_one_robot_per_terrain(env)
        slugs = [TERRAIN_SLUGS[tid] for tid in order]
    obs, obs_hist = env.get_observations()
    j = JOINT_INDEX
    scale = float(env.cfg.control.action_scale)

    vx_cmd = 1.0
    warmup = int(round(2.0 / env.dt))
    n_log = int(round(8.0 / env.dt))
    n = len(slugs)
    keys = [
        'dof_pos_target', 'dof_pos', 'dof_vel', 'dof_torque',
        'command_x', 'command_y', 'command_yaw',
        'base_vel_x', 'base_vel_y', 'base_vel_z', 'base_vel_yaw',
        'contact_forces_z',
    ]
    buf = {k: [[] for _ in range(n)] for k in keys}
    alive = [True] * n
    prev_ep = env.episode_length_buf[:n].detach().cpu().numpy().copy()

    for step in range(warmup + n_log):
        env.commands[:n, 0] = vx_cmd
        env.commands[:n, 1] = 0.0
        env.commands[:n, 2] = 0.0
        actions = policy(obs.detach(), obs_hist.detach())
        step_out = env.step(actions.detach())
        obs, obs_hist = step_out[0], step_out[3]
        if step < warmup:
            prev_ep = env.episode_length_buf[:n].detach().cpu().numpy().copy()
            continue
        ep = env.episode_length_buf[:n].detach().cpu().numpy()
        sample = {
            'dof_pos_target': (actions[:n, j] * scale).detach().cpu().numpy(),
            'dof_pos': env.dof_pos[:n, j].detach().cpu().numpy(),
            'dof_vel': env.dof_vel[:n, j].detach().cpu().numpy(),
            'dof_torque': env.torques[:n, j].detach().cpu().numpy(),
            'command_x': env.commands[:n, 0].detach().cpu().numpy(),
            'command_y': env.commands[:n, 1].detach().cpu().numpy(),
            'command_yaw': env.commands[:n, 2].detach().cpu().numpy(),
            'base_vel_x': env.base_lin_vel[:n, 0].detach().cpu().numpy(),
            'base_vel_y': env.base_lin_vel[:n, 1].detach().cpu().numpy(),
            'base_vel_z': env.base_lin_vel[:n, 2].detach().cpu().numpy(),
            'base_vel_yaw': env.base_ang_vel[:n, 2].detach().cpu().numpy(),
            'contact_forces_z': env.contact_forces[:n][:, env.feet_indices, 2].detach().cpu().numpy(),
        }
        for i in range(n):
            if not alive[i]:
                continue
            if ep[i] < prev_ep[i]:
                alive[i] = False
                continue
            for k in keys:
                buf[k][i].append(np.array(sample[k][i], copy=True))
        prev_ep = ep.copy()

    out_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'terrain_logs')
    os.makedirs(out_dir, exist_ok=True)
    stale = ('vel', 'angle', 'force')
    for i, slug in enumerate(slugs):
        n_pts = len(buf['base_vel_x'][i])
        if n_pts < 2:
            print(f'[log] {slug} 没有有效样本，跳过')
            continue
        for name in (f'{slug}.png',) + tuple(f'{slug}_{s}.png' for s in stale):
            old = os.path.join(out_dir, name)
            if os.path.isfile(old):
                os.remove(old)
        log = {k: np.asarray(buf[k][i], dtype=np.float64) for k in keys}
        t = np.arange(n_pts, dtype=np.float64) * float(env.dt)
        for path in save_logger_rows(out_dir, slug, t, log):
            print(f'[log] saved {path}  steps={n_pts}')


if __name__ == '__main__':
    play(get_args(), mesh=os.environ.get('AMP_LOG_MESH', 'trimesh'))
