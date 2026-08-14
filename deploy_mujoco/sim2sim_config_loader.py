"""
Sim2Sim 配置加载器：直接从文件路径加载配置类，绕过 legged_gym/envs/__init__.py，
避免在 sim2sim 推理时加载 isaacgym（sim2sim 只需 mujoco + torch + 配置参数）。

用法（在 sim2sim 脚本里）：
    from sim2sim_config_loader import load_cfg
    GO2_Trot_Cfg_Yu = load_cfg(
        "legged_gym/envs/Go2_MoB/GO2_Trot/GO2_Trot_config.py",
        "GO2_Trot_Cfg_Yu",
    )
"""
import importlib.util
import sys
import os
from legged_gym import LEGGED_GYM_ROOT_DIR


def load_cfg(config_rel_path, class_name):
    """从相对仓库根的路径加载配置类，不触发 envs/__init__.py。

    Args:
        config_rel_path: 配置文件相对仓库根的路径，如 "legged_gym/envs/.../xxx_config.py"。
        class_name: 要加载的配置类名，如 "GO2_Trot_Cfg_Yu"。

    Returns:
        配置类（未实例化）。
    """
    config_abs_path = os.path.join(LEGGED_GYM_ROOT_DIR, config_rel_path)
    if not os.path.exists(config_abs_path):
        raise FileNotFoundError(f"配置文件不存在: {config_abs_path}")

    # 生成唯一模块名，避免冲突
    module_name = f"_sim2sim_cfg_{os.path.basename(config_rel_path).replace('.py', '')}"

    # 直接从文件加载模块（不走包导入，不触发 envs/__init__.py）
    spec = importlib.util.spec_from_file_location(module_name, config_abs_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"配置文件 {config_rel_path} 中没有类 {class_name}")

    return getattr(module, class_name)
