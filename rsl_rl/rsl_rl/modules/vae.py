import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, 
                cenet_in_dim, #vae的输入维度
                num_latent_dim, #隐变量的维度
                num_est_dims, #显式估计的变量的维度
                num_decode_dims=45, #解码器的输出维度
                activation="elu", 
                **kwargs):
        if kwargs:
            print("ActorCritic_DWAQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(VAE, self).__init__()
        self.activation = get_activation(activation)

        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim,128),
            self.activation,
            nn.Linear(128,64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64,num_latent_dim)
        self.encode_logvar_latent = nn.Sequential(
            nn.Linear(64,num_latent_dim),
            nn.Hardtanh(min_val=-5., max_val=5.) # to avoid numerical issues
            )
        self.encode_mean_vel = nn.Linear(64,num_est_dims)
        self.encode_logvar_vel =nn.Sequential(
            nn.Linear(64,num_est_dims),
            nn.Hardtanh(min_val=-5., max_val=5.) # to avoid numerical issues
            )


        self.decoder = nn.Sequential(
            nn.Linear(num_latent_dim+num_est_dims,128),
            self.activation,
            nn.Linear(128,128),
            self.activation,
            nn.Linear(128,num_decode_dims)
        )
    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        return mean + var*code_temp
    
    def cenet_forward(self,obs_history):
        encoded = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(encoded)
        logvar_latent = self.encode_logvar_latent(encoded)
        mean_vel = self.encode_mean_vel(encoded)
        logvar_vel = self.encode_logvar_vel(encoded)
        code_latent = self.reparameterise(mean_latent,logvar_latent)
        code_vel = self.reparameterise(mean_vel,logvar_vel)
        
        code = torch.cat((code_vel,code_latent),dim=-1)
        decode = self.decoder(code)
        return (code),(code_vel,code_latent),(decode),(mean_vel,logvar_vel,mean_latent,logvar_latent)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None