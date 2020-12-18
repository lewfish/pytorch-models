# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
def sample_gmm(pis, mvns, N):
    all_samples = []
    all_sample_inds = []
    for ind, pi in enumerate(pis):
        comp_N = int(round(N * pi.item()))
        samples = mvns[ind].sample((comp_N,))
        all_samples.append(samples)
        sample_inds = torch.full((comp_N,), ind, dtype=torch.int)
        all_sample_inds.append(sample_inds)

    return torch.cat(all_samples), torch.cat(all_sample_inds)

from matplotlib.patches import Ellipse

# From https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(sample_inds, samples, mvns=None):
    plt.scatter(samples[:, 0], samples[:, 1], c=sample_inds)
    if mvns is not None:
        for mvn in mvns:
            draw_ellipse(
                mvn.mean.detach().numpy(),
                mvn.covariance_matrix.detach().numpy(),
                alpha=0.2)
    plt.legend()
    plt.show()

# %%
def sample_dataset1(N):
    mus = torch.tensor([
        [-10., -2],
        [0., 0.],
        [2, 2]
    ])
    sigmas = torch.tensor([
        [[5., 2], [2., 2]],
        [[1., 0], [0., 1]],
        [[0.1, 0], [0, 0.5]],
    ])
    mvns = [
        MultivariateNormal(mus[0], sigmas[0]),
        MultivariateNormal(mus[1], sigmas[1]),
        MultivariateNormal(mus[2], sigmas[2]),
    ]
    pis = torch.tensor([0.1, 0.4, 0.5])
    samples, sample_inds = sample_gmm(pis, mvns, N)
    return samples, sample_inds

def sample_moons(N):
    samples, sample_inds = make_moons(N, noise=0.1)
    samples, sample_inds = torch.tensor(samples, dtype=torch.float32), torch.tensor(sample_inds)
    return samples, sample_inds

N = 1000
samples, sample_inds = sample_moons(N)
# samples, sample_inds = sample_dataset1(N)
scaler = StandardScaler()
scaler.fit(samples)
samples = torch.tensor(scaler.transform(samples), dtype=torch.float32)
plot_gmm(sample_inds, samples)

# %%
class GmmSgd(nn.Module):
    def __init__(self, k, d):
        super(GmmSgd, self).__init__()
        self.k = k
        self.d = d

        self.mus = nn.Parameter(torch.tensor(torch.randn((k, d)), requires_grad=True))
        self.pi_params = nn.Parameter(torch.ones((k,), requires_grad=True))
        sigma_params = torch.eye(d).unsqueeze(0).repeat(k, 1, 1)
        self.sigma_params = nn.Parameter(
            torch.tensor(sigma_params, requires_grad=True))

    def get_dists(self):
        pis = torch.nn.functional.softmax(self.pi_params, dim=0)
        mvns = [
            MultivariateNormal(
                self.mus[i],
                torch.matmul(self.sigma_params[i], self.sigma_params[i].transpose(0, 1)))
            for i in range(self.k)]
        return pis, mvns

    def __str__(self):
        pis, mvns = self.get_dists()
        summary = [f'pis: {pis}']
        for i in range(self.k):
            summary.append(f'\ncomponent {i}')
            summary.append(f'mu: {mvns[i].loc}')
            summary.append(f'sigma: {mvns[i].covariance_matrix}')
        return '\n'.join(summary)

    def forward(self, X):
        pis, mvns = self.get_dists()
        probs = [mvns[i].log_prob(X).exp().unsqueeze(1) for i in range(self.k)]
        probs = torch.cat(probs, dim=1)
        neg_log_like = -(probs * pis).sum(dim=1).log().mean()
        return neg_log_like

    def summarize(self, N=1000):
        print(str(self))
        pis, mvns = self.get_dists()
        samples, sample_inds = sample_gmm(pis, mvns, N)
        plot_gmm(sample_inds, samples, mvns)

# %%
d = 2
k = 10
model = GmmSgd(k, d)

batch_sz = 100
lr = 1e-2
num_epochs = 100
log_interval = 25

optimizer = optim.Adam(model.parameters(), lr=lr)
train_ds = TensorDataset(samples)
train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

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

model.summarize()

# %%
class GmmEm():
    def __init__(self, k, d):
        self.k = k
        self.d = d

        self.mus = torch.tensor(torch.randn((k, d)))
        self.pis = torch.ones((k,)) / self.k
        self.sigmas = torch.eye(d).unsqueeze(0).repeat(k, 1, 1)

    def get_mvns(self):
        return [
            MultivariateNormal(self.mus[i], self.sigmas[i])
            for i in range(self.k)]

    def __str__(self):
        mvns = self.get_mvns()
        summary = [f'pis: {self.pis}']
        for i in range(self.k):
            summary.append(f'\ncomponent {i}')
            summary.append(f'mu: {mvns[i].loc}')
            summary.append(f'sigma: {mvns[i].covariance_matrix}')
        return '\n'.join(summary)

    def get_comp_probs(self, X):
        mvns = self.get_mvns()
        probs = [mvns[i].log_prob(X).exp().unsqueeze(1) for i in range(self.k)]
        return torch.cat(probs, dim=1)

    def get_neg_log_like(self, X):
        comp_probs = self.get_comp_probs(X)
        return -(comp_probs * self.pis).sum(dim=1).log().mean()

    def compute_resps(self, X):
        probs = self.get_comp_probs(X)
        return probs / probs.sum(dim=1, keepdims=True)

    def update_params(self, X, resps):
        self.pis = resps.mean(dim=0)

        for k in range(self.k):
            self.mus[k] = (X * resps[:, k].unsqueeze(1)).sum(dim=0) / resps[:, k].sum()
            X_ = (X - self.mus[k].unsqueeze(0)) * resps[:, k].unsqueeze(1)
            self.sigmas[k] = torch.matmul(X_.transpose(1, 0), X_) / resps[:, k].sum()

    def em_step(self, X):
        resps = self.compute_resps(X)
        self.update_params(X, resps)
        return self.get_neg_log_like(X)

    def summarize(self, N=1000):
        print(str(self))
        mvns = self.get_mvns()
        samples, sample_inds = sample_gmm(self.pis, mvns, N)
        plot_gmm(sample_inds, samples, mvns)

# %%
d = 2
k = 10
model = GmmEm(k, d)

num_iters = 10
train_ds = TensorDataset(samples)
train_dl = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)

for iter in range(num_iters):
    X, = list(train_dl)[0]
    neg_log_like = model.em_step(X)
    print(f'neg_log_like={neg_log_like:.3f}')

model.summarize()

# %%
skgmm = GaussianMixture(n_components=k, verbose=1)
skgmm.fit(samples)
s = skgmm.sample(N)
plot_gmm(s[1], s[0])
