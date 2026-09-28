# On-policy runner for Go2_PIE (parkour_mjlab PIE logic).

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO_PIE
from rsl_rl.modules import ActorCriticPIE
from rsl_rl.env import VecEnv


class PIERunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        depth_shape = tuple(self.env.cfg.camera.output_hw)
        # (C, H, W) with C = image_nums
        depth_chw = (
            int(self.env.cfg.camera.image_nums or 1),
            int(depth_shape[0]),
            int(depth_shape[1]),
        )
        history_dim = self.env.cfg.env.num_history_obs
        height_map_dim = int(self.env.cfg.env.num_height_map)
        foot_dim = int(self.env.cfg.env.num_foot_clearance)
        if num_critic_obs != self.env.num_obs + 3 + height_map_dim:
            print(
                f"[PIERunner] critic dim={num_critic_obs}, expected "
                f"{self.env.num_obs}+3+{height_map_dim}="
                f"{self.env.num_obs + 3 + height_map_dim}"
            )

        actor_critic = ActorCriticPIE(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            history_dim=history_dim,
            depth_shape=depth_chw,
            velocity_dim=3,
            height_map_dim=height_map_dim,
            foot_dim=foot_dim,
            **self.policy_cfg,
        ).to(self.device)
        actor_critic.init_hidden(self.env.num_envs, self.device)

        self.alg = PPO_PIE(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        depth_flat = depth_chw[0] * depth_chw[1] * depth_chw[2]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [history_dim],
            [depth_flat],
            [3],
            [height_map_dim],
            [foot_dim],
            [self.env.num_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, obs_hist, depth = self.env.get_observations()
        privileged_obs, pie_targets = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        obs_hist = obs_hist.to(self.device)
        depth = depth.to(self.device)
        velocity_target = pie_targets["velocity"].to(self.device)
        height_target = pie_targets["height"].to(self.device)
        foot_target = pie_targets["foot"].to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(
                        obs,
                        critic_obs,
                        obs_hist,
                        depth,
                        velocity_target,
                        height_target,
                        foot_target,
                    )
                    (
                        obs,
                        privileged_obs,
                        obs_hist,
                        depth,
                        pie_targets,
                        rewards,
                        dones,
                        infos,
                    ) = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    obs_hist = obs_hist.to(self.device)
                    depth = depth.to(self.device)
                    velocity_target = pie_targets["velocity"].to(self.device)
                    height_target = pie_targets["height"].to(self.device)
                    foot_target = pie_targets["foot"].to(self.device)
                    successor = pie_targets["successor"].to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos, successor)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_entropy_loss,
                mean_auxiliary,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "terrain" in key:
                    self.writer.add_scalar("Terrain/" + key, value, locs["it"])
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/entropy", locs["mean_entropy_loss"], locs["it"])
        for name, value in locs["mean_auxiliary"].items():
            self.writer.add_scalar(f"Loss/pie_{name}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )

        aux = locs["mean_auxiliary"]
        aux_str = " ".join("{}={:.4f}".format(k, v) for k, v in aux.items())
        title = " Learning iteration {}/{} ".format(
            locs["it"],
            self.current_learning_iteration + locs["num_learning_iterations"],
        )
        log_string = (
            f"{'#' * width}\n"
            f"{title:=^{width}}\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
            f"{'Entropy:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"
            f"{'PIE aux:':>{pad}} {aux_str}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.current_learning_iteration = loaded["iter"]
        return loaded.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)

        def policy(obs, obs_hist, depth):
            actions, _ = self.alg.actor_critic.act_inference(obs, obs_hist, depth)
            return actions

        return policy
