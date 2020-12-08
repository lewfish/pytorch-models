# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import torch.nn as nn

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
def sample_gmm(mus, sigmas, pis, N):
    all_samples = []
    all_sample_inds = []
    for ind, pi in enumerate(pis):
        comp_N = int(round(N * pi.item()))
        samples = (torch.randn((comp_N,)) * sigmas[ind]) + mus[ind]
        all_samples.append(samples)
        sample_inds = torch.full((comp_N,), ind, dtype=torch.int)
        all_sample_inds.append(sample_inds)

    return torch.cat(all_samples), torch.cat(all_sample_inds)

def plot_gmm(samples):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().numpy()
    plt.hist(samples, bins=30)
    plt.show()

# %%

if False:
    mus = torch.tensor([-20., 0., 200])
    sigmas = torch.tensor([0.5, 3.0, 0.1])
    pis = torch.tensor([0.1, 0.4, 0.5])

if True:
    mus = torch.tensor([-2., 0., 2])
    sigmas = torch.tensor([0.5, 3.0, 0.1])
    pis = torch.tensor([0.1, 0.4, 0.5])

gt_params = [mus, sigmas, pis]
N = 1000
k = 3

samples, sample_inds = sample_gmm(mus, sigmas, pis, N)
plot_gmm(samples)

# %%
import math

def normal_prob(mu, sigma, x):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

x = torch.linspace(-10., 10, 100)
y = normal_prob(0., 1., x)
plt.plot(x, y)

# %%

def log_normal_prob(mu, sigma, x):
    return (torch.log(torch.tensor((1.0 / (sigma * math.sqrt(2.0 * math.pi))))) +
            (-0.5 * ((x - mu) / sigma) ** 2))

x = torch.linspace(-10., 10, 100)
y = log_normal_prob(0., 1., x)
plt.plot(x, y)
plt.show()

plt.plot(x, y.exp())
plt.show()

# %%

x = torch.randn((3, 3), requires_grad=True)
lse = x.exp().sum(dim=1).log().mean()
lse2 = torch.logsumexp(x, dim=1).mean()

print(lse)
print(lse2)

lse.backward()
print(x.grad)

x.grad.zero_()
print(x.grad)
lse2.backward()
print(x.grad)

# %%

class GMM(nn.Module):
    def __init__(self, k):
        super(GMM, self).__init__()
        self.k = k
        self.mus = nn.Parameter(torch.randn((k,), requires_grad=True))
        self.sigma_params = nn.Parameter(torch.full((k,), 0.0, requires_grad=True))
        self.pi_params = nn.Parameter(torch.ones((k,), requires_grad=True))

    def get_dists(self):
        pis = torch.nn.functional.softmax(self.pi_params, dim=0)
        sigmas = self.sigma_params.exp()
        return self.mus, sigmas, pis

    def forward(self, X):
        mus, sigmas, pis = self.get_dists()
        probs = torch.cat([normal_prob(mus[i], sigmas[i], X).unsqueeze(1) for i in range(k)], dim=1)
        neg_log_like = -(probs * pis).sum(dim=1).log().mean()
        # log_probs = torch.cat([log_normal_prob(mus[i], sigmas[i], X).unsqueeze(1) for i in range(k)], dim=1)
        # neg_log_like = -torch.logsumexp(log_probs + pis.log(), dim=1).mean()
        return neg_log_like


# %%

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

model = GMM(k)
batch_sz = 100
lr = 1e-2
num_epochs = 301
log_interval = 100

sample_mu = samples.mean()
sample_std = samples.std()
norm_samples = (samples - sample_mu) / sample_std

optimizer = optim.Adam(model.parameters(), lr=lr)
train_ds = TensorDataset(norm_samples)
train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

all_mus = []
all_sigmas = []
all_pis = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (X,) in enumerate(train_dl):
        optimizer.zero_grad()
        loss = model(X)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch % log_interval == 0:
        print(f'epoch: {epoch} / loss: {running_loss}')
    running_loss = 0.0

    mus, sigmas, pis = model.get_dists()
    all_mus.append(mus.data.clone().unsqueeze(0) * sample_std + sample_mu)
    all_sigmas.append(sigmas.clone().unsqueeze(0) * sample_std)
    all_pis.append(pis.data.clone().unsqueeze(0))

# %%

mus, sigmas, pis = model.get_dists()
model_samples, _ = sample_gmm(mus, sigmas, pis, N)
model_samples = (model_samples * sample_std) + sample_mu
plot_gmm(model_samples)

mus = torch.cat(all_mus).detach()
sigmas = torch.cat(all_sigmas).detach()
pis = torch.cat(all_pis).detach()
params = [mus, sigmas, pis]
titles = ['mus', 'sigmas', 'pis']

fig, axs = plt.subplots(len(params), 1, figsize=(len(params), 3 * len(params)))
for pi, p in enumerate(params):
    for ki in range(k):
        axs[pi].set_ylabel(titles[pi])
        axs[pi].axhline(gt_params[pi][ki], linestyle='--', alpha=0.2)
        axs[pi].plot(torch.arange(0, len(p)), p[:, ki])

# %%
from sklearn.mixture import GaussianMixture

skgmm = GaussianMixture(n_components=3, verbose=1)
skgmm.fit(samples.unsqueeze(1))
s = skgmm.sample(300)
plot_gmm(s[0])

# %%
