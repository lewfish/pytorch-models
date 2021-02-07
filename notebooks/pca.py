# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
N = 50
theta = torch.linspace(0, 2*math.pi, N)
x = torch.cos(theta)
y = torch.sin(theta)
xy = torch.stack([x, y], dim=0)

# A = torch.randn((2, 2))
# A = torch.matmul(A, A.transpose(0, 1))
A = torch.tensor(
    [[ 0.5066, -0.6642],
    [-0.6642,  1.3764]])
u, s, v = torch.svd(A)
eig = torch.eig(A, eigenvectors=True)

vxy = torch.matmul(v.transpose(1, 0), xy)
svxy = torch.matmul(torch.diag(s), vxy)
usvxy = torch.matmul(u, svxy)
Axy = torch.matmul(A, xy)
points = [xy, vxy, svxy, usvxy, Axy]

max_val = torch.cat(points, dim=1).abs().flatten().max()

# %%
fig, axs = plt.subplots(1, len(points), figsize=(3 * len(points), 3))
for i in range(len(points)):
    axs[i].scatter(points[i][0], points[i][1], c=theta)
    axs[i].set_xlim(-max_val, max_val)
    axs[i].set_ylim(-max_val, max_val)

# %%
fix, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(Axy[0], Axy[1], c=theta)
ax.scatter(xy[0], xy[1], c=theta)
ev = eig.eigenvectors[:, 0]
ax.plot([0, ev[0]], [0, ev[1]])
ev = eig.eigenvectors[:, 1]
ax.plot([0, ev[0]], [0, ev[1]])
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)

# %%
from torch.distributions import MultivariateNormal
mvn = MultivariateNormal(torch.zeros((2,)), A)
samples = mvn.sample((300,))

fix, axs = plt.subplots(1, 3, figsize=(9, 3))
ax = axs[0]
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1)

ax = axs[1]
u, s, v = torch.svd(samples)
recon = torch.chain_matmul(u, torch.diag(s), v.transpose(1, 0))
ax.scatter(recon[:, 0], recon[:, 1], alpha=0.1)
ev = v[:, 0]
ax.plot([0, ev[0]], [0, ev[1]], color='red')
ev = v[:, 1]
ax.plot([0, ev[0]], [0, ev[1]], color='red')

ax = axs[2]
u, s, v = torch.svd_lowrank(samples, q=1)
recon = torch.chain_matmul(u, torch.diag(s), v.transpose(1, 0))
ax.scatter(recon[:, 0], recon[:, 1], alpha=0.1)

for ax in axs:
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

# %%
# transform high-dim to latent space and then back again
# least squares can be solved as follows due to the fact that the basis is orthonormal
u, s, v = torch.svd(samples)
new_samples = mvn.sample((300,))
latent = torch.matmul(new_samples, v)
recon = torch.matmul(latent, v.transpose(1, 0))

fix, axs = plt.subplots(1, 3, figsize=(9, 3))

ax = axs[0]
ax.scatter(new_samples[:, 0], new_samples[:, 1], alpha=0.1)
ax = axs[1]
ax.scatter(latent[:, 0], latent[:, 1], alpha=0.1)
ax = axs[2]
ax.scatter(recon[:, 0], recon[:, 1], alpha=0.1)

for ax in axs:
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

# %%
from sklearn.datasets import load_sample_image
im = load_sample_image('china.jpg')
im = np.sum(im.astype(np.float), axis=2) / (3 * 255)
plt.imshow(im, cmap='gray')

# %%
scaler = StandardScaler()
scaler.fit(im.transpose())
X = torch.tensor(scaler.transform(im.transpose()), dtype=torch.float32).transpose(1, 0)
u, s, v = torch.svd_lowrank(X, 30)
recon = torch.chain_matmul(u, torch.diag(s), v.transpose(1, 0))
plt.imshow(scaler.inverse_transform(recon.numpy().transpose()).transpose(), cmap='gray')
