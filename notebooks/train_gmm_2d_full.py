# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.mixture import GaussianMixture

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

def plot_gmm(sample_inds, samples):
    plt.scatter(samples[:, 0], samples[:, 1], c=sample_inds)
    plt.legend()
    plt.show()

# %%

if True:
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
    N = 1000

    k = 3
    mus = mus[0:k]
    mvns = mvns[0:k]
    pis = pis[0:k]

if False:
    mus = torch.tensor([
        [0., 0.],
    ])
    sigmas = torch.tensor([
        [[1., 0], [0., 1]],
    ])
    mvns = [
        MultivariateNormal(mus[0], sigmas[0]),
    ]
    pis = torch.tensor([1.0])
    N = 1000

samples, sample_inds = sample_gmm(pis, mvns, N)
plot_gmm(sample_inds, samples)

sample_mu = samples.mean(dim=0)
sample_std = samples.std(dim=0)
norm_samples = (samples - sample_mu) / sample_std

# %%

class GMM(nn.Module):
    def __init__(self, k, d, mus=None, learn_pis=True, learn_sigma=True):
        super(GMM, self).__init__()
        self.k = k
        self.d = d

        if mus is None:
            mus = torch.randn((k, d))
        self.mus = nn.Parameter(torch.tensor(mus, requires_grad=True))

        if learn_pis:
            self.pi_params = nn.Parameter(torch.ones((k,), requires_grad=True))
        else:
            self.pi_params = torch.ones((k,))

        sigma_params = torch.eye(d).unsqueeze(0).repeat(k, 1, 1)
        if learn_sigma:
            self.sigma_params = nn.Parameter(
                torch.tensor(sigma_params, requires_grad=True))
        else:
            self.sigma_params = sigma_params

    def get_dists(self):
        pis = torch.nn.functional.softmax(self.pi_params, dim=0)
        mvns = [
            MultivariateNormal(
                self.mus[i],
                torch.matmul(self.sigma_params[i], self.sigma_params[i].transpose(0, 1)))
            for i in range(k)]
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
        probs = [mvns[i].log_prob(X).exp().unsqueeze(1) for i in range(k)]
        probs = torch.cat(probs, dim=1)
        neg_log_like = -(probs * pis).sum(dim=1).log().mean()
        return neg_log_like

    def summarize(self, N=1000):
        print(str(self))
        pis, mvns = self.get_dists()
        samples, sample_inds = sample_gmm(pis, mvns, N)
        samples = (samples + sample_mu) * sample_std
        plot_gmm(sample_inds, samples)

# %%

d = 2
k = 3
learn_pis = True
learn_sigma = True
model = GMM(k, d, learn_pis=learn_pis, learn_sigma=learn_sigma)

batch_sz = 100
lr = 1e-2
num_epochs = 301
plot_interval = 100

optimizer = optim.Adam(model.parameters(), lr=lr)
train_ds = TensorDataset(norm_samples)
train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (X,) in enumerate(train_dl):
        optimizer.zero_grad()
        loss = model(X)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch % plot_interval == 0:
        print(f'epoch: {epoch} / loss: {running_loss}')

    running_loss = 0.0

model.summarize()

# %%

skgmm = GaussianMixture(n_components=3, verbose=1)
skgmm.fit(samples)
s = skgmm.sample(N)
plot_gmm(s[1], s[0])

# %%
