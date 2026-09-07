# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# DreamWaQ-main runner + AMP discriminator loop.

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO_DreamWaQ_AMP
from rsl_rl.modules import ActorCriticDreamWaQ
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer


class DreamWaQRunner_AMP:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        num_critic_obs = (
            self.env.num_privileged_obs
            if self.env.num_privileged_obs is not None
            else self.env.num_obs
        )
        cenet_in_dim = self.env.cfg.env.num_history_obs
        cenet_out_dim = self.env.cfg.env.num_latent_dims + self.env.cfg.env.num_explicit_dims

        actor_critic = ActorCriticDreamWaQ(
            self.env.num_obs + cenet_out_dim,
            num_critic_obs,
            self.env.num_actions,
            cenet_in_dim,
            cenet_out_dim,
            decode_dim=self.env.num_obs,
            **self.policy_cfg,
        ).to(self.device)

        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["runner"]["amp_num_preload_transitions"],
            motion_files=self.cfg["amp_motion_files"],
            amp_obs_dim=24,  # 关节角12 + 关节角速度12
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg["runner"]["amp_reward_coef"],
            train_cfg["runner"]["amp_discr_hidden_dims"],
            device,
        ).to(self.device)
        min_std = torch.tensor(self.cfg["min_normalized_std"], device=self.device) * (
            torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])
        )

        self.alg = PPO_DreamWaQ_AMP(
            actor_critic,
            discriminator,
            amp_data,
            amp_normalizer,
            min_std,
            device=self.device,
            num_obs=self.env.num_obs,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [num_critic_obs],
            [cenet_in_dim],
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

        obs, obs_hist = self.env.get_observations()
        privileged_obs, prev_critic_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        prev_critic_obs = prev_critic_obs.to(self.device)
        obs_hist = obs_hist.to(self.device)
        amp_obs = self.env.get_amp_observations().to(self.device)
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, prev_critic_obs, obs_hist, amp_obs)
                    (
                        obs,
                        privileged_obs,
                        prev_privileged_obs,
                        obs_hist,
                        rewards,
                        dones,
                        infos,
                        reset_env_ids,
                        terminal_amp_states,
                    ) = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    prev_critic_obs = prev_privileged_obs
                    next_amp_obs = self.env.get_amp_observations()

                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    prev_critic_obs = prev_critic_obs.to(self.device)
                    obs_hist = obs_hist.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    reset_env_ids = reset_env_ids.to(self.device)
                    terminal_amp_states = terminal_amp_states.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
                    rewards, _ = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        rewards,
                        normalizer=self.alg.amp_normalizer,
                    )
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_vae_loss,
                mean_vel_estimation_loss,
                mean_recon_loss,
                mean_kl_loss,
                mean_amp_loss,
                mean_grad_pen_loss,
                mean_policy_pred,
                mean_expert_pred,
                mean_entropy_loss,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
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
                elif "max_command" in key:
                    self.writer.add_scalar("CommandCurriculum/" + key, value, locs["it"])
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/entropy", locs["mean_entropy_loss"], locs["it"])
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/vae_function", locs["mean_vae_loss"], locs["it"])
        self.writer.add_scalar("Loss/mean_kl_loss", locs["mean_kl_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/mean_vel_estimation_loss", locs["mean_vel_estimation_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/mean_recon_loss", locs["mean_recon_loss"], locs["it"])
        self.writer.add_scalar("Loss/AMP", locs["mean_amp_loss"], locs["it"])
        self.writer.add_scalar("Loss/AMP_grad", locs["mean_grad_pen_loss"], locs["it"])
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
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str_title = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                f"""{'vae function loss:':>{pad}} {locs['mean_vae_loss']:.4f}\n"""
                f"""{'Vel Estimation loss:':>{pad}} {locs['mean_vel_estimation_loss']:.4f}\n"""
                f"""{'Recon loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                f"""{'VAE KL Divergence Loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                f"""{'vae function loss:':>{pad}} {locs['mean_vae_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "discriminator_state_dict": self.alg.discriminator.state_dict(),
                "amp_normalizer": self.alg.amp_normalizer,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                print(f"[DreamWaQRunner_AMP] skip optimizer load (architecture changed): {e}")
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
