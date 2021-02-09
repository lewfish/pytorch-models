# Based on https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
# %%
from IPython import get_ipython
from IPython.display import HTML

import math

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal, Categorical
from pyvirtualdisplay import Display
import gym

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
env_name = 'CartPole-v0'
env = gym.make(env_name)
env.reset()

# %%
def demo_agent(env, model=None, max_steps=None):
    _display = Display(visible=False, size=(800, 800))
    _ = _display.start()

    fig, ax = plt.subplots()
    obs = env.reset()
    img = ax.imshow(env.render(mode='rgb_array'))

    frames = []
    while True:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        if model is None:
            action = env.action_space.sample()
        else:
            obs = torch.from_numpy(obs).float()
            action = sample_action(model, obs)

        obs, reward, done, info = env.step(action)
        if done:
            break
        if max_steps is not None and len(frames) >= max_steps:
            break

    def animate(frame):
        img.set_data(frame)
    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=50, frames=frames)
    return ani

# %%
def make_block(inp, out):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.ReLU()
    )

def make_mlp(*sizes):
    layers = []
    for i in range(len(sizes)-2):
        layers.append(make_block(sizes[i], sizes[i+1]))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)

def sample_action(model, obs):
    action_logits = model(obs)
    action_dist = Categorical(logits=action_logits)
    return action_dist.sample().item()

# %%
def get_trajectory(model, env, max_steps=None):
    obs_lst = []
    reward_lst = []
    action_lst = []

    with torch.no_grad():
        obs = env.reset()
        step = 0

        while True:
            obs = torch.from_numpy(obs).float()
            obs_lst.append(obs.unsqueeze(0))
            action = sample_action(model, obs)
            action_lst.append(action)
            obs, reward, done, info = env.step(action)
            reward_lst.append(reward)

            if done:
                break

            step += 1

            if max_steps is not None and step >= max_steps:
                break

    return torch.cat(obs_lst), torch.tensor(action_lst), torch.tensor(reward_lst)

def reward2go(rewards):
    return rewards.flip(0).cumsum(0).flip(0)

def get_batch(model, env, batch_sz):
    obs_lst = []
    action_lst = []
    r2g_lst = []
    rsum_lst = []
    curr_sz = 0
    nb_trajs = 0

    while True:
        max_steps = batch_sz - curr_sz
        obs, actions, rewards = get_trajectory(model, env, max_steps)

        r2g = reward2go(rewards)
        rsum = rewards.sum().item()
        rsum_lst.append(rsum)

        obs_lst.append(obs)
        action_lst.append(actions)
        r2g_lst.append(r2g)
        curr_sz += len(r2g)
        nb_trajs += 1

        if curr_sz >= batch_sz:
            break

    rmean = torch.tensor(rsum_lst).mean()

    return torch.cat(obs_lst), torch.cat(action_lst), torch.cat(r2g_lst), rmean

def get_loss(model, obs, actions, r2g):
    action_logits = model(obs)
    action_dist = Categorical(logits=action_logits)
    action_log_probs = action_dist.log_prob(actions)
    return -(action_log_probs * r2g).mean()

def train(env, model, num_epochs=100, batch_sz=5000, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        obs, actions, r2g, rmean = get_batch(model, env, batch_sz)
        loss = get_loss(model, obs, actions, r2g)
        loss.backward()
        optimizer.step()

        print(f'epoch: {epoch} / mean reward: {rmean:.2f}')

# %%
obs_sz = env.observation_space.shape[0]
nb_actions = env.action_space.n
mlp = make_mlp(obs_sz, 128, nb_actions)
train(env, mlp, num_epochs=50, batch_sz=5000)

# %%
ani = demo_agent(env, mlp)
HTML(ani.to_jshtml())

# %%
