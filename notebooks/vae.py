# %%
from IPython import get_ipython

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
N = 300
# samples, sample_inds = make_moons(N, noise=0.1)
x = torch.linspace(-5, 5, N)
y = torch.sin(x) + torch.rand_like(x)
samples = torch.stack([x, y], dim=1)
scaler = StandardScaler()
scaler.fit(samples)
samples = torch.tensor(scaler.transform(samples), dtype=torch.float32)
plt.scatter(samples[:, 0], samples[:, 1])

# %%
class VAE(nn.Module):
    def __init__(self, x_sz, z_sz, hidden_sz=100):
        super(VAE, self).__init__()
        self.x_sz = x_sz
        self.z_sz = z_sz
        hidden_sz = hidden_sz

        self.enc_hidden = nn.Sequential(
            nn.Linear(x_sz, hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, hidden_sz),
            nn.ReLU())
        self.enc_mu = nn.Linear(hidden_sz, z_sz)
        self.enc_logvar = nn.Linear(hidden_sz, z_sz)

        self.dec_hidden = nn.Sequential(
            nn.Linear(z_sz, hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, hidden_sz),
            nn.ReLU())
        self.dec_mu = nn.Linear(hidden_sz, x_sz)
        self.dec_logvar = nn.Linear(hidden_sz, x_sz)

    def encode(self, X):
        out = self.enc_hidden(X)
        mu = self.enc_mu(out)
        logvar = self.enc_logvar(out)
        return mu, logvar

    def sample_latent(self, mu, logvar):
        noise = torch.randn(mu.shape)
        std = torch.exp(logvar / 2)
        return mu + noise * std

    def sample_obs(self, N):
        z = torch.randn(N, self.z_sz)
        return self.dec_mu(self.dec_hidden(z))

    def reconstruct(self, X):
        mu, _ = self.encode(X)
        return mu, self.dec_mu(self.dec_hidden(mu))

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.sample_latent(mu, logvar)

        out = self.dec_hidden(z)
        x_mu = self.dec_mu(out)
        x_logvar = self.dec_logvar(out)
        return mu, logvar, x_mu, x_logvar

    def loss(self, X, mu, logvar, x_mu, x_logvar, batch_ratio, train_progress):
        mvn = Normal(x_mu, x_logvar.exp().unsqueeze(0))
        recon = -torch.mean(mvn.log_prob(X))
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + kld
        return {'loss': loss, 'recon': recon, 'kld': kld}

# %%
x_sz = 2
z_sz = 2
hidden_sz = 1000
model = VAE(x_sz, z_sz, hidden_sz)

batch_sz = 100
lr = 1e-3
num_epochs = 100
log_interval = 1

optimizer = optim.Adam(model.parameters(), lr=lr)
train_ds = TensorDataset(samples)
train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
batch_ratio = batch_sz / len(train_ds)

recon = torch.zeros(num_epochs)
kld = torch.zeros(num_epochs)

for epoch in range(num_epochs):
    running_loss = 0.0
    train_progress = epoch / num_epochs

    for i, (X,) in enumerate(train_dl):
        optimizer.zero_grad()
        mu, logvar, x_mu, x_logvar = model(X)
        loss_dict = model.loss(X, mu, logvar, x_mu, x_logvar, batch_ratio, train_progress)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        recon[epoch] += loss_dict['recon'].item()
        kld[epoch] += loss_dict['kld'].item()

    if epoch % log_interval == 0:
        print(f'epoch: {epoch} / loss: {running_loss}')

    running_loss = 0.0

# %%
model.eval()
with torch.no_grad():
    mu, recon_samples = model.reconstruct(samples)
    gen_samples = model.sample_obs(N)

plt.plot(torch.arange(num_epochs), recon, label='recon')
plt.plot(torch.arange(num_epochs), kld, label='kld')
plt.legend()
plt.show()

plt.scatter(samples[:, 0], samples[:, 1])
plt.title('training samples')
plt.show()

plt.scatter(recon_samples[:, 0], recon_samples[:, 1])
plt.title('reconstructed samples')
plt.show()

plt.scatter(mu[:, 0], mu[:, 1])
plt.title('latents')
plt.show()

fig, axs = plt.subplots(z_sz, 1, figsize=(3, 3 * z_sz))
for ind, ax in enumerate(axs):
    ax.hist(mu[:, ind].numpy())
fig.suptitle('latents')
plt.show()

plt.scatter(gen_samples[:, 0], gen_samples[:, 1])
plt.title('generated samples')
plt.show()

# %%
