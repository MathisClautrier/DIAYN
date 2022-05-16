import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

def weight_init(m):
    """Custom weight init Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

class QNet(nn.Module):
    def __init__(self,action_shp,observation_shp,hidden_layers):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(observation_shp+action_shp,hidden_layers[0]))
        for i in range(1,len(hidden_layers)):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_layers[i-1],hidden_layers[i]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers[-1],1))
        self.layers = nn.Sequential(*self.layers)

        self.apply(weight_init)

    def forward(self,observation,action):
        assert observation.size(0) == action.size(0)

        obs_action = torch.cat([observation, action], dim=1)
        return self.layers(obs_action)

class Critic(nn.Module):
    def __init__(self,action_shp,observation_shp,hidden_layers=(300,300)):
        super().__init__()


        self.Q1 = QNet(action_shp,observation_shp,hidden_layers)
        self.Q2 = QNet(action_shp,observation_shp,hidden_layers)

    def forward(self,observation,action):
        q1 = self.Q1(observation,action)
        q2 = self.Q2(observation,action)
        return q1,q2

class Actor(nn.Module):
    def __init__(self,observation_shp,action_shp,num_skills,log_std_min,
    log_std_max,hidden_layers=(300,300)):
        super().__init__()
        self.layers = []
        print(observation_shp,action_shp,num_skills,hidden_layers)
        self.layers.append(nn.Linear(observation_shp + num_skills,hidden_layers[0]))
        for i in range(1,len(hidden_layers)):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_layers[i-1],hidden_layers[i]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers[-1],2*action_shp))
        self.layers = nn.Sequential(*self.layers)

        self.num_skills = num_skills
        self.log_std_M = log_std_max
        self.log_std_m = log_std_min

        self.apply(weight_init)

    def forward(self,obs,z):
        if len(z.shape)==2:
            z = z.squeeze(-1)
        skills = F.one_hot(z.long(),num_classes = self.num_skills)
        inputs = torch.cat([obs,skills],dim=1)
        mu,log_std = self.layers(inputs).chunk(2,dim=-1)
        log_std = self.log_std_m + 0.5 * (self.log_std_M - self.log_std_m) * (torch.tanh(log_std) + 1)
        std = torch.exp(log_std)
        epsilon = torch.randn_like(mu)
        pi = epsilon*std + mu

        log_pi = (-0.5 * epsilon.pow(2) - log_std).sum(-1, keepdim=True) - 0.5 * np.log(2 * np.pi) * epsilon.size(-1)

        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        log_pi = log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)

        return mu,log_std,pi,log_pi

class Discriminator(nn.Module):
    def __init__(self,observation_shp,num_skills,hidden_layers=(300,300)):
        super().__init__()
        self.observation_shp = observation_shp
        self.num_skills = num_skills

        self.layers = []
        self.layers.append(nn.Linear(observation_shp,hidden_layers[0]))
        for i in range(1,len(hidden_layers)):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_layers[i-1],hidden_layers[i]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers[-1],self.num_skills))
        self.layers.append(nn.LogSoftmax(dim=-1))
        self.layers = nn.Sequential(*self.layers)

        self.apply(weight_init)

    def forward(self,observation,skill,select=True):
        if select:
            if not type(skill) == torch.Tensor:
                skill = torch.tensor(skill)
            if len(skill.shape)==1:
                skill = skill.unsqueeze(-1)
            outputs = self.layers(observation)
            return outputs.gather(1,skill.type(torch.int64))
        else:
            return self.layers(observation)
        
