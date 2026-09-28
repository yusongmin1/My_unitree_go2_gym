"""Go2_PIE 策略加载器（sim2sim 用，不经 jit）。

ActorCriticPIE 的推理路径带内部 GRU 状态（memory_hidden 随每次
act_inference 滚动），jit trace 无法保存这种副作用，因此 sim2sim
直接在 python 里构建网络并加载 checkpoint。

用法：
    from pie_policy_loader import load_pie_policy
    policy = load_pie_policy("logs/go2_pie/xxx/model_200.pt")  # cpu, eval
    actions, _ = policy(obs(1,45), obs_hist(1,450), depth(1,10320))
注意：本模块必须在本文件顶部导入（其内部会先触发 isaacgym，再导 torch）。
"""
from sim2sim_config_loader import load_cfg  # noqa: E402  必须在 torch 前

_cfg_cls = load_cfg("legged_gym/envs/Go2_PIE/Go2_PIE_Config.py", "Go2_PIE_Cfg_Yu")
_cfg_ppo = load_cfg("legged_gym/envs/Go2_PIE/Go2_PIE_Config.py", "Go2_PIE_PPO_Yu")

import torch  # noqa: E402

from rsl_rl.modules.actor_critic_pie import ActorCriticPIE  # noqa: E402


def load_pie_policy(checkpoint_path, device="cpu"):
    """构建 ActorCriticPIE 并加载 checkpoint，返回 eval 模式的推理策略。

    返回的模块每次调用 act_inference 会滚动内部 memory_hidden，
    与训练 rollout 行为一致；batch 维固定 1。
    """
    p = _cfg_ppo.policy
    e = _cfg_cls.env
    c = _cfg_cls.camera
    depth_chw = (int(c.image_nums or 1), int(c.output_hw[0]), int(c.output_hw[1]))

    ac = ActorCriticPIE(
        int(e.num_single_obs),
        int(e.num_single_obs) + 3 + int(e.num_height_map),  # critic 维，推理不用
        int(e.num_actions),
        history_dim=int(e.num_history_obs),
        depth_shape=depth_chw,
        velocity_dim=3,
        height_map_dim=int(e.num_height_map),
        foot_dim=int(e.num_foot_clearance),
        init_noise_std=p.init_noise_std,
        actor_hidden_dims=list(p.actor_hidden_dims),
        critic_hidden_dims=list(p.critic_hidden_dims),
        activation=p.activation,
        token_dim=p.token_dim,
        map_latent_dim=p.map_latent_dim,
        vae_latent_dim=p.vae_latent_dim,
        memory_hidden_dim=p.memory_hidden_dim,
        memory_num_layers=p.memory_num_layers,
        history_hidden_dims=list(p.history_hidden_dims),
        successor_decoder_dims=list(p.successor_decoder_dims),
        height_decoder_dims=list(p.height_decoder_dims),
        output_channels=list(p.output_channels),
        kernel_size=list(p.kernel_size),
        stride=list(p.stride),
        padding=p.padding,
        attention_heads=p.attention_heads,
        transformer_layers=p.transformer_layers,
        transformer_ff_dim=p.transformer_ff_dim,
        transformer_dropout=p.transformer_dropout,
        cnn_activation=p.cnn_activation,
    ).to(device)

    ck = torch.load(checkpoint_path, map_location=device)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    missing, unexpected = ac.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint 有多余权重: {unexpected[:5]}")
    bad = [k for k in missing
           if any(s in k for s in ("actor", "encoder", "memory.weight", "memory.bias", "head"))]
    if bad:
        raise RuntimeError(f"关键权重缺失: {bad}")

    ac.eval()
    ac.init_hidden(1, device)
    print(f"[pie_policy_loader] 已加载 {checkpoint_path}"
          f" (iter={ck.get('iter', '?') if isinstance(ck, dict) else '?'}), missing={len(missing)}")
    return ac
