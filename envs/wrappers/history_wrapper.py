from gym.core import Env
import isaacgym
assert isaacgym
import torch
import gym
from base.legged_robot_config import LeggedRobotCfg

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.obs_history_length = self.env.LeggedRobotCfg.env.num_obs_hist
        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float, device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs


    def step(self,actions):
        obs,rew,done,info = self.env.step(actions)
        priveleged_obs = info["priveleged_obs"]
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': priveleged_obs, 'obs_history': self.obs_history}, rew, done, info
    

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}
    

    def reset_idx(self, env_ids):
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret
    

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}
