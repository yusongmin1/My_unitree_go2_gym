"""HIMLoco 式地形隐空间可视化（t-SNE）。

论文 Fig.4：机器人在各类地形上行走时，记录 history encoder 的隐变量
（不含显式速度），用真实地形类别上色，再做 t-SNE 降到 2D。

本脚本对应四个任务：
  go2_dreamwaq / go2_amp_dreamwaq  → CENet mean_latent（16 维，不用估计速度）
  go2_cts / go2_amp_cts            → student encoder（本体历史，部署时实际用的那支）

无 viewer，结束直接写 PNG。

  python legged_gym/scripts/pca_terrain_latent.py --task=go2_dreamwaq --headless
"""
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
import matplotlib
matplotlib.use('Agg')
from legged_gym.envs import *  # noqa: F401,F403  注册任务
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch

# 与 terrain.make_terrain 的 idx 对齐（proportions 为增量，内部会先做 cumsum）
TERRAIN_NAMES = {
    0: '平缓斜坡',
    1: '粗糙斜坡',
    2: '下楼梯',
    3: '上楼梯',
    4: '离散障碍',
    5: '跨沟',
    6: '踏石',
    7: '双排踏石',
    8: '单排踏石',
    9: '独木桥',
    10: '跨栏',
    11: '坡道',
    12: '半楼梯',
}
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#393b79', '#637939', '#8c6d31',
]


def _setup_font():
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for name in ['Noto Serif CJK SC', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'SimHei']:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [name]
            break
    plt.rcParams['axes.unicode_minus'] = False


def column_terrain_idx(num_cols, proportions):
    """每个地形列的类别，逻辑与 Terrain.make_terrain 的 choice 分支一致。"""
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


def tsne_2d(x):
    from sklearn.manifold import TSNE
    x = np.asarray(x, dtype=np.float64)
    perplexity = min(30.0, max(5.0, (len(x) - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=0,
    ).fit_transform(x)


def _model(runner):
    if hasattr(runner.alg, 'actor_critic'):
        return runner.alg.actor_critic
    return runner.alg.model


@torch.no_grad()
def encode_latent(model, obs_hist):
    """只取隐变量：DreamWaQ 去掉 3 维速度；CTS 用 student（本体历史）。"""
    if hasattr(model, 'encode_mean_latent'):
        h = model.encoder(obs_hist)
        return model.encode_mean_latent(h)
    if hasattr(model, 'student_encoder'):
        return model.student_encoder(obs_hist)
    raise RuntimeError('当前策略没有 CENet / student encoder，无法做地形隐空间可视化')


def _split_obs(pack, family):
    if family == 'dreamwaq':
        return pack[0], pack[1]
    return pack[0], pack[2]


def _split_step(out, family):
    if family == 'dreamwaq':
        return out[0], out[3]
    return out[0], out[2]


def run(args):
    import matplotlib.pyplot as plt
    _setup_font()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 200
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 3
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg.domain_rand, 'push_towards_goal'):
        env_cfg.domain_rand.push_towards_goal = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e9
    if hasattr(env_cfg.env, 'test'):
        env_cfg.env.test = True

    args.headless = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 固定中等难度，避免课程把机器人换到别的行
    level = min(3, env.cfg.terrain.num_rows - 1)
    env.terrain_levels[:] = level
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]

    got = env.get_observations()
    family = 'dreamwaq' if len(got) == 2 else 'cts'
    env.reset()
    got = env.get_observations()
    obs, hist = _split_obs(got, family)

    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    model = _model(runner)
    model.eval()
    # runner 构造时会再 reset 一次，观测以这次为准
    obs, hist = _split_obs(env.get_observations(), family)

    col_idx = column_terrain_idx(env.cfg.terrain.num_cols, env.cfg.terrain.terrain_proportions)
    env_idx = col_idx[env.terrain_types.detach().cpu().numpy()]

    warmup, collect, stride = 150, 250, 2
    latents, labels = [], []
    for i in range(warmup + collect):
        env.commands[:, 0] = 0.5
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        actions = policy(obs.detach(), hist.detach())
        out = env.step(actions.detach())
        obs, hist = _split_step(out, family)
        if i < warmup or (i - warmup) % stride != 0:
            continue
        # 站立样本：重力投影 z 接近 -1；并跳过复位后的自由落体窗口
        upright = env.projected_gravity[:, 2] < -0.4
        ready = env.episode_length_buf > 50
        keep = (upright & ready).detach().cpu().numpy()
        if not np.any(keep):
            continue
        z = encode_latent(model, hist.detach())
        latents.append(z.detach().cpu().numpy()[keep])
        labels.append(env_idx[keep])

    if not latents:
        raise RuntimeError(f'{args.task}: 没有采集到站立样本，检查策略是否还能站住')

    x = np.concatenate(latents, axis=0)
    y = np.concatenate(labels, axis=0)
    # 每类最多 600 点，避免某一类把图铺满
    rng = np.random.default_rng(0)
    keep_rows = []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        if len(ids) > 600:
            ids = rng.choice(ids, 600, replace=False)
        keep_rows.append(ids)
    keep_rows = np.concatenate(keep_rows)
    x, y = x[keep_rows], y[keep_rows]

    z2 = tsne_2d(x)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for c in sorted(np.unique(y)):
        m = y == c
        name = TERRAIN_NAMES.get(int(c), f'地形{int(c)}')
        ax.scatter(
            z2[m, 0], z2[m, 1],
            s=12, alpha=0.55, linewidths=0,
            c=COLORS[int(c) % len(COLORS)],
            label=f'{name} (n={int(m.sum())})',
        )
    ax.set_xlabel('t-SNE-1')
    ax.set_ylabel('t-SNE-2')
    ax.set_title(f'{args.task}  隐变量 t-SNE')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=10, markerscale=1.6, framealpha=0.9)
    fig.tight_layout()

    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tsne_terrain_latent.png')
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f'[t-SNE] {args.task} family={family} latent_dim={x.shape[1]} samples={len(y)}')
    print(f'[t-SNE] saved {out_path}')


if __name__ == '__main__':
    run(get_args())
