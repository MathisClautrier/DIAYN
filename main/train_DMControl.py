
import numpy as np
import torch
import os
import random
import time
import json
import dmc2gym
import wandb
from argparse import Namespace
import numpy as np

from storage import ReplayBuffer

from agent import DIAYN


def evaluate(env, agent, num_episodes, step, args):
    all_ep_rewards = []

    def run_eval_loop():
        start_time = time.time()
        for i in range(num_episodes):
            obs = env.reset()
            z=np.random.choice(agent.num_skills)
            z = torch.tensor([z])
            done = False
            episode_reward = 0
            agent.actor.eval()
            n=0
            while not done:
                action,_ = agent.select_action(obs,z)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                if n > 100:
                    break
                n+=1

            wandb.log({'eval_episode_reward': episode_reward}, step)
            all_ep_rewards.append(episode_reward)
            agent.actor.train()
        
        wandb.log({'eval_eval_time': time.time()-start_time }, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        wandb.log({'eval_mean_episode_reward': mean_ep_reward}, step)
        wandb.log({'eval_best_episode_reward': best_ep_reward}, step)

    run_eval_loop()

def train(cfg):
    with open(cfg) as f:
        args_dic = json.load(f)
    args = Namespace(**args_dic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=False,
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = args.work_dir + '/'  + exp_name


    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("work on :",device)

    action_shape = env.action_space.shape

    obs_shape = env.observation_space.shape

    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
    )
    num_skills = args['num_skills']
    agent = DIAYN(obs_shape,action_shape,device,num_skills,**args_dic)


    episode, episode_reward, done = 0, 0, True
    z=np.random.choice(agent.num_skills)
    #        z = torch.tensor([z])

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            evaluate(env, agent, args.num_eval_episodes,  step,args)

        if done:
            if step % args.log_interval == 0:
                wandb.log({'train/episode_reward': episode_reward}, step)

            obs = env.reset()
            z = np.random.choice(agent.num_skills)
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:

            action = agent.sample_action(obs,torch.tensor([z]))


        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, step)

        next_obs, reward, done, _ = env.step(action)
        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, z, done_bool)

        obs = next_obs
        episode_step += 1

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    train("cfg/cfg.json")