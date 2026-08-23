# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import torch.nn as nn
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # Sort by modification time (most recent first)
        runs.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)), reverse=True)
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[0])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        # 按文件名中的数字排序（字典序会让 900 > 1700，导致永远加载 900 轮的模型）
        import re as _re
        models.sort(key=lambda m: int(_re.findall(r'\d+', m)[-1]) if _re.findall(r'\d+', m) else 0)
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "a1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--teacher", "action": "store_true", "default": False, "help": "Play with teacher policy (e.g. play_amp_cts); export still uses student."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
class PolicyExporterCTS(torch.nn.Module):
    def __init__(self, actor_critic, ):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.student_encoder = copy.deepcopy(actor_critic.student_encoder)
        self.actor.eval()
        self.student_encoder.eval()
    def forward(self, obs_history):
        latent= self.student_encoder(obs_history[:,:-45])
        actor_input = torch.cat(
            (latent,obs_history[:, -45:]), dim=1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, 'policy_cts.pt')
        self.to('cpu')
        self.eval()
        with torch.no_grad():
            traced_script_module = torch.jit.script(self)
            traced_script_module.save(export_path)
            print(f"模型已导出至: {export_path}")


def export_policy_as_cts(actor_critic, path, ):
    exporter = PolicyExporterCTS(actor_critic)
    exporter.export(path)
class PolicyExporterDWAQ(torch.nn.Module):
    def __init__(self, actor_critic, ):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
        self.actor.eval()
        # self.vae.eval()
    def forward(self, obs_history):
        (code),(code_vel,code_latent),(decode),(mean_vel,logvar_vel,mean_latent,logvar_latent)= self.vae.cenet_forward(obs_history[:,:-45])
        actor_input = torch.cat(
            (code,obs_history[:, -45:]), dim=1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, 'policy_dwaq.pt')
        self.to('cpu')
        self.eval()
        with torch.no_grad():
            traced_script_module = torch.jit.script(self)
            traced_script_module.save(export_path)
            print(f"模型已导出至: {export_path}")


def export_policy_as_dwaq(actor_critic, path, ):
    if hasattr(actor_critic, 'vae'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterDWAQ(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

# import os
# import copy
# import torch
# import torch.onnx

# class PolicyExporterDWAQ(torch.nn.Module):
#     def __init__(self, actor_critic):
#         super().__init__()
#         self.actor = copy.deepcopy(actor_critic.actor)
#         self.vae = copy.deepcopy(actor_critic.vae)
#         self.actor.eval()
#         self.vae.eval()          # 必须 eval 模式

#     def forward(self, obs_history):
#         # 与训练时完全一致
#         (code,), *_ = self.vae.cenet_forward(obs_history[:, :-57])
#         actor_input = torch.cat((code.unsqueeze(0), obs_history[:, -57:]), dim=1)
#         actions_mean = self.actor(actor_input)
#         return actions_mean

#     # ==========  改写为 ONNX  ==========
#     def export(self, path, opset=11):
#         os.makedirs(path, exist_ok=True)
#         onnx_file = os.path.join(path, 'policy_dwaq.onnx')

#         # 1. 构造 dummy 输入（batch=1，长度与你训练时一致）
#         seq_len = 285 +57                # 请改成你真实的 obs_history 总长度
#         dummy_obs = torch.randn(1, seq_len)

#         # 2. 切换到 CPU、eval、无梯度
#         self.to('cpu')
#         self.eval()
#         with torch.no_grad():
#             torch.onnx.export(
#                 self,                                    # 模型
#                 (dummy_obs,),                            # 输入 tuple
#                 onnx_file,
#                 input_names=['obs_history'],
#                 output_names=['actions_mean'],
#                 dynamic_axes={                           # batch 维度可变
#                     'obs_history': {0: 'batch'},
#                     'actions_mean': {0: 'batch'}
#                 },
#                 opset_version=opset
#             )
#         print(f'ONNX 模型已导出至: {onnx_file}')


# def export_policy_as_dwaq(actor_critic, path):
#     if hasattr(actor_critic, 'vae'):
#         exporter = PolicyExporterDWAQ(actor_critic)
#         exporter.export(path)
#     else:
#         # 不含 VAE 的分支也顺手改成 ONNX
#         os.makedirs(path, exist_ok=True)
#         onnx_file = os.path.join(path, 'policy_1.onnx')
#         model = copy.deepcopy(actor_critic.actor).cpu().eval()
#         dummy = torch.randn(1, actor_critic.actor_input_dim)  # 换成真实 dim
#         with torch.no_grad():
#             torch.onnx.export(
#                 model, (dummy,), onnx_file,
#                 input_names=['obs'],
#                 output_names=['actions'],
#                 dynamic_axes={'obs': {0: 'batch'}, 'actions': {0: 'batch'}},
#                 opset_version=11
#             )
#         print(f'ONNX 模型已导出至: {onnx_file}')


class LSTMEncoderAMP(nn.Module):
    def __init__(self,input_size=45,hidden_size=256,num_layers=3) -> None:
        super().__init__()
        self.LSTM = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
      
        encoder_dims = [256,128]
        self.output_dim = 16 + 16
        mlp_layers = []
        mlp_layers.append(nn.Linear(256,encoder_dims[0]))
        mlp_layers.append(nn.ELU())
        for l in range(len(encoder_dims)):
           if l == len(encoder_dims) -1:
              mlp_layers.append(nn.Linear(encoder_dims[l],self.output_dim))
           else:
              mlp_layers.append(nn.Linear(encoder_dims[l],encoder_dims[l+1]))
              mlp_layers.append(nn.ELU())
        self.output = nn.Sequential(*mlp_layers)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    
    def forward(self,obs,h,c):
        scan_latent, (he,ce) = self.LSTM(obs[:, None, :], (h,c))
        scan_latent = self.output(scan_latent.squeeze(1))
        return scan_latent,he,ce

from copy import deepcopy

class PolicyExporterJit(nn.Module):
    def __init__(self, lstm_encoder, actor):
        super().__init__()
        self.lstm_encoder = LSTMEncoderAMP()
        self.lstm_encoder.load_state_dict(lstm_encoder.state_dict(), strict=True)
        self.actor = deepcopy(actor)

        # 注册 h 和 c 为 buffer，初始为空张量
        self.register_buffer("h", torch.zeros(0))
        self.register_buffer("c", torch.zeros(0))
        # 如果 h/c 是空张量，初始化为 0
        if self.h.numel() == 0 or self.c.numel() == 0:
            self.h = torch.zeros(3, 1, 256, device=torch.device("cpu"))
            self.c = torch.zeros(3, 1, 256,  device=torch.device("cpu"))
    def forward(self, obs):
        batch_size = obs.shape[0]
        device = obs.device
        scan_latent, self.h, self.c = self.lstm_encoder(obs, self.h, self.c)
        backbone_input = torch.cat([scan_latent, obs], dim=1)
        actions = self.actor(backbone_input)
        return actions

# ============ 导出函数 ============
def export_policy_as_jit_amp(lstm_encoder, actor, path):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, "policy_amp_ts.pt")

    device = torch.device("cpu")
    policy = PolicyExporterJit(lstm_encoder, actor).to(device).eval()

    # 构造假输入
    num_envs = 1
    dummy_obs = torch.ones(num_envs, 45, device=device)
    dummy_h   = torch.zeros(3, num_envs, 256, device=device)
    dummy_c   = torch.zeros(3, num_envs, 256, device=device)

    with torch.no_grad():
        traced = torch.jit.script(policy)
        traced.save(save_path)

    print(f"[JIT] Policy exported -> {os.path.abspath(save_path)}")


class PolicyExporterOnnx(torch.nn.Module):
    def __init__(self, lstm_encoder,actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic)
        self.lstm_encoder = LSTMEncoderAMP()
        self.lstm_encoder.load_state_dict(lstm_encoder.state_dict(), strict=True)

    def forward(self,obs,h,c):
        scan_latent,he,ce = self.lstm_encoder.forward(obs,h,c)
        backbone_input = torch.cat([scan_latent,obs], dim=1)
        actions = self.actor(backbone_input)
        return actions,he,ce

def export_policy_as_onnx(lstm_encoder, actor_critic, path):
    policy = PolicyExporterOnnx(lstm_encoder, actor_critic)
    device = torch.device('cpu')
    policy = policy.to(device)#.cpu()
    policy.eval()
    
    with torch.no_grad():
        if not os.path.exists(os.path.join(path, "traced")):
            os.mkdir(os.path.join(path, "traced"))
        save_path = os.path.join(path,"traced","policy_amp_ts.onnx")
        num_envs = 1
        dummy_input = torch.ones(num_envs, 45, device=device)
        h = torch.zeros(3, 1, 256, device=device)  # LSTM 初始 h
        c = torch.zeros(3, 1, 256, device=device)  # LSTM 初始 c
        torch.onnx.export(
            policy, 
            (dummy_input, h, c),  # 这里直接用 h, c
            save_path, 
            input_names=["obs", "h", "c"],  # 这里改回 h, c
            output_names=["actions", "he", "ce"],
            opset_version=11
        )    
        print(f"Model exported to {save_path}")        
        print("Saved traced_actor at ", os.path.abspath(save_path))
