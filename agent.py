
from models import Actor,Critic, Discriminator

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

class DIAYN(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        num_skills,
        device,
        hidden_layer=[256,256],
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        log_interval=100,
        discri_lr = 1e-4,
        **kwargs
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.num_skills = num_skills
        
        self.actor = Actor(obs_shape,action_shape, num_skills,
            actor_log_std_min, actor_log_std_max, hidden_layers=hidden_layer
            ).to(device)

        self.critic = Critic(
            action_shape,obs_shape,hidden_layers = hidden_layer
        ).to(device)

        self.critic_target = Critic(
            action_shape,obs_shape,hidden_layers = hidden_layer
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.discriminator = Discriminator(obs_shape,num_skills,hidden_layers = hidden_layer).to(device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(),
            lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.proj.parameters())+list(self.critic.Q1.parameters())+list(self.critic.Q2.parameters()), 
            lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.discriminator_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(),
            lr = discri_lr,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.critic_target.train()
        self.critic.train()
        self.actor.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs,z):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs,z)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs,z):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            _,_, pi, _  = self.actor(obs,z)
            return pi.cpu().data.numpy().flatten()

    def critic_step(self,obs,action,reward,next_obs,skills,not_done,step):
        with torch.no_grad():
            _,_,pi,log_pi=self.actor(next_obs,skills)
            target_Q1, target_Q2 = self.critic_target(next_obs, pi)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            diayn_reward = self.discriminator(obs,skills) - \
            torch.log(torch.ones_like(skills)/self.num_skills)
            target_Q = diayn_reward + (not_done * self.discount * target_V)
        
        current_Q1, current_Q2 = self.critic(
            obs, action)

        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            wandb.log({'train_critic_loss': critic_loss}, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def actor_alpha_step(self,obs,skills,step):
        _,_,pi,log_pi,_ = self.actor(obs,skills)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            wandb.log({'train_actor_loss': actor_loss}, step)
            #wandb.log({'train_actor_target_entropy': self.target_entropy}, step)
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            wandb.log({'train_alpha/loss': alpha_loss}, step)
            wandb.log({'train_alpha/value': self.alpha}, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def discriminator_step(self,obs,skills,step):
        predicted_skills = self.discriminator(obs,skills,select=False)
        one_hot = F.one_hot(skills,num_classes = self.num_skills)

        discriminator_loss = self.cross_entropy_loss(predicted_skills,one_hot.float())

        if step %self.log_interval ==0:
            wandb.log({'train_discriminator_loss':discriminator_loss},step)
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def update(self,replay_buffer,step):
        obs, action, reward, next_obs, skills, not_done = replay_buffer.sample()

    
        if step % self.log_interval == 0:
            wandb.log({'train/batch_reward': reward.mean()}, step)

        self.critic_step(obs, action, reward, next_obs, not_done, skills, step)
        self.discriminator_step(obs,skills,step)
        if step % self.actor_update_freq == 0:
            self.actor_alpha_step(obs, skills, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )