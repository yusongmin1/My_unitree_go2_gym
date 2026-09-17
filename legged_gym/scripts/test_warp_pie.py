"""Go2_PIE + warp 深度相机迁移冒烟测试（小规模，不训练）。"""
import isaacgym  # noqa
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args

args = get_args()
args.num_envs = 36
args.headless = True

env, env_cfg = task_registry.make_env(name="go2_pie", args=args)

print("=" * 60)
print("[1] env 创建成功：")
print("    image_buf:", None if env.image_buf is None else tuple(env.image_buf.shape))
assert not hasattr(env, "image_clean_buf"), "image_clean_buf 应已删除"
assert env.image_buf.shape == (36, 2, 60, 86), f"image_buf shape 异常: {env.image_buf.shape}"

print("[2] 跑 12 个 policy step：")
for _ in range(12):
    env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))

img = env.image_buf
print("[3] image_buf 统计：")
print("    shape:", tuple(img.shape), "dtype:", img.dtype)
print("    min/max/mean: %.4f / %.4f / %.4f" % (img.min(), img.max(), img.mean()))
assert img.min() >= 0.0 and img.max() <= 1.0, "归一化越界！"
assert (img == 0).float().mean() > 0.001, "dropout 噪声未生效"

env0 = img[0, -1].cpu().numpy()
print("[4] env0 最新帧：中心行 mean=%.3f | 顶行 mean=%.3f" % (env0[29].mean(), env0[2].mean()))
print("[5] 双帧差异 mean=%.5f" % (img[:, 0] - img[:, 1]).abs().mean())
print("=" * 60)
print("全部检查通过 ✓")
