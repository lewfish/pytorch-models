# Based on https://d2l.ai/chapter_generative-adversarial-networks/gan.html

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
from torch.distributions import MultivariateNormal

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
mu = torch.tensor([0., 0.])
sigma = torch.tensor([
    [1.0100, 1.9500],
    [1.9500, 4.2500]])
X = MultivariateNormal(mu, sigma).sample((1000,))
plt.scatter(X[:, 0], X[:, 1])

# %%
latent_sz = 2
gen = nn.Sequential(
    nn.Linear(latent_sz, 2),
)
disc = nn.Sequential(
    nn.Linear(2, 5),
    nn.Tanh(),
    nn.Linear(5, 3),
    nn.Tanh(),
    nn.Linear(3, 1),
)

for p in gen.parameters():
    nn.init.normal_(p, 0, 0.02)
for p in disc.parameters():
    nn.init.normal_(p, 0, 0.02)

# %%
def plot_generator(n=1000):
    with torch.no_grad():
        z = torch.normal(0, 1, (n, latent_sz))
        fake_x = gen(z).numpy()
        plt.scatter(fake_x[:, 0], fake_x[:, 1])
        plt.title('generated samples')
        plt.show()

plot_generator()

# %%
nb_epochs = 20
batch_sz = 8
gen_lr = 0.005
disc_lr = 0.05

gen.train()
disc.train()
train_ds = TensorDataset(X)
train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)

gen_opt = optim.Adam(gen.parameters(), lr=gen_lr)
disc_opt = optim.Adam(disc.parameters(), lr=disc_lr)
loss_fn = nn.BCEWithLogitsLoss()
gen_losses = []
disc_losses = []

for epoch in range(nb_epochs):
    gen_loss = 0.
    disc_loss = 0.

    for real_x in train_dl:
        real_x = real_x[0]
        curr_batch_sz = real_x.shape[0]

        # update disc: try to improve at predicting real vs fake
        disc_opt.zero_grad()
        gen_opt.zero_grad()

        z = torch.normal(0, 1, (curr_batch_sz, latent_sz))
        fake_x = gen(z)
        fake_y = disc(fake_x)
        real_y = disc(real_x)

        y = torch.cat([fake_y, real_y])
        target = torch.cat([torch.zeros(curr_batch_sz), torch.ones(curr_batch_sz)])
        loss = loss_fn(y[:, 0], target)
        disc_loss += loss.item()
        loss.backward()
        disc_opt.step()

        # update gen: try to get discriminator to predict that fakes are real
        gen_opt.zero_grad()
        disc_opt.zero_grad()
        z = torch.normal(0, 1, (curr_batch_sz, latent_sz))
        fake_x = gen(z)
        fake_y = disc(fake_x)
        target = torch.ones(curr_batch_sz)
        loss = loss_fn(fake_y[:, 0], target)
        gen_loss += loss.item()
        loss.backward()
        gen_opt.step()

    gen_loss = gen_loss / len(train_ds)
    gen_losses.append(gen_loss)
    disc_loss = disc_loss / len(train_ds)
    disc_losses.append(disc_loss)
    print(f'epoch: {epoch} / gen_loss: {gen_loss:.4f} / disc_loss: {disc_loss:.4f}')

disc_losses = np.array(disc_losses)
gen_losses = np.array(gen_losses)

# %%
disc.eval()
gen.eval()
def plot_losses():
    plt.plot(np.arange(len(disc_losses)), disc_losses, label='disc')
    plt.plot(np.arange(len(gen_losses)), gen_losses, label='gen')
    plt.legend()
    plt.title('losses')
    plt.show()

plot_losses()
plot_generator()

# %%
