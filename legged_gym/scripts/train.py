import os
import shutil
import inspect

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # 把当前 env / config 源码备份到本次 log_dir，便于复现
    log_dir = ppo_runner.log_dir
    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        task_class = task_registry.get_task_class(args.task)
        robot_file_path = inspect.getfile(task_class)
        robot_config_path = inspect.getfile(type(env_cfg))
        shutil.copy(robot_file_path, log_dir)
        shutil.copy(robot_config_path, log_dir)
        # PPO/train cfg 若在单独文件也一并备份
        train_config_path = inspect.getfile(type(train_cfg))
        if os.path.abspath(train_config_path) != os.path.abspath(robot_config_path):
            shutil.copy(train_config_path, log_dir)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
