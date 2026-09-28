"""导出 Go2_PIE 策略为 torch.jit（sim2sim 用），GRU hidden 显式输入/输出。

ActorCriticPIE 的 rollout 路径会把 GRU 状态存在 self.memory_hidden 里（副作用），
jit.trace 无法保存。这里包装成纯函数：

    actions, hidden_next = forward(obs(1,45), obs_hist(1,450), depth(1,10320), hidden(1,1,128))

sim2sim 侧持有 hidden 张量逐步回传，语义与训练 rollout 完全一致。

用法（仓库根目录，amp 环境）:
    python deploy_mujoco/export_pie_jit.py \
        --checkpoint logs/go2_pie/Sep28_18-57-01_/model_100.pt \
        [--output legged_gym/envs/Go2_PIE/exported/policies/policy_pie.pt]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pie_policy_loader import load_pie_policy, _cfg_cls, _cfg_ppo  # noqa: E402  最先导入（isaacgym 先于 torch）

import argparse  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402


class PiePolicyJit(nn.Module):
    """PIE 策略纯函数包装：hidden 显式输入/输出，可 trace。"""

    def __init__(self, ac):
        super().__init__()
        self.ac = ac
        e = _cfg_cls.env
        self.num_obs = int(e.num_single_obs)
        c = _cfg_cls.camera
        self.depth_dim = int(c.image_nums or 1) * int(c.output_hw[0]) * int(c.output_hw[1])

    def forward(self, obs, obs_hist, depth, hidden):
        # 等价 act_inference 的 rollout 路径，但 hidden 由外部传入、更新值返回
        estimates, new_hidden = self.ac._estimate_tensors(
            obs, obs_hist, depth,
            hidden_state=hidden,
            update_internal=False,   # 不动内部状态
            sample_latent=False,     # 推理用均值
        )
        actor_input = torch.cat([obs, estimates["latent"]], dim=-1)
        return self.ac.actor(actor_input), new_hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str,
                        default=f'{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/Go2_PIE/exported/policies/policy_pie.pt')
    args = parser.parse_args()

    ac = load_pie_policy(args.checkpoint)
    wrapper = PiePolicyJit(ac).eval()

    e = _cfg_cls.env
    c = _cfg_cls.camera
    hdim = int(_cfg_ppo.policy.memory_hidden_dim)
    obs = torch.zeros(1, int(e.num_single_obs))
    hist = torch.zeros(1, int(e.num_history_obs))
    depth = torch.zeros(1, int(c.image_nums or 1) * int(c.output_hw[0]) * int(c.output_hw[1]))
    hidden = torch.zeros(1, 1, hdim)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (obs, hist, depth, hidden), strict=False)

        # 一致性验证：trace vs eager，滚动 5 步 hidden 对比
        ac.init_hidden(1, 'cpu')  # eager 路径用内部状态
        torch.manual_seed(0)
        o, hh, dd = obs.clone(), hist.clone(), depth.clone()
        hu = hidden.clone()
        max_diff = 0.0
        for _ in range(5):
            dd = torch.rand_like(depth)  # 每步换 depth 模拟滚动输入
            a_t, hu = traced(o, hh, dd, hu)
            a_e, _ = ac.act_inference(o, hh, dd)  # 内部滚动 memory_hidden
            max_diff = max(max_diff, (a_t - a_e).abs().max().item())
        print(f'一致性检查: trace vs eager 最大差 {max_diff:.2e} (期望 <1e-5)')
        assert max_diff < 1e-4, 'trace 与 eager 输出不一致！'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    traced.save(args.output)
    print(f'saved -> {args.output}')
    print(f'调用方式: actions, hidden = policy(obs(1,{obs.shape[1]}), hist(1,{hist.shape[1]}), '
          f'depth(1,{depth.shape[1]}), hidden(1,1,{hdim}))')


if __name__ == '__main__':
    main()
