# %%
from IPython import get_ipython

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, Categorical
import torch.nn as nn

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
class CondDist():
    """p(x|y)"""
    def log_prob(self, x, y):
        pass

    def sample(self, y, shape):
        pass

class Dist():
    """p(x)"""
    def log_prob(self, x):
        pass

    def sample(self, shape):
        pass

def plot(x, z, x_hat_mean=None, x_hat_std=None):
    T = x.shape[0]
    time = torch.arange(0, T, 1)
    plt.plot(time, x, label='x')
    plt.plot(time, z, label='z')
    if x_hat_mean is not None:
        plt.plot(time, x_hat_mean, label='x_hat')
    if x_hat_std is not None:
        plt.gca().fill_between(
            time, x_hat_mean + x_hat_std, x_hat_mean - x_hat_std, alpha=0.4)
    plt.legend()
    plt.show()

class SSM():
    def __init__(self, px0, pxx, pzx):
        self.px0 = px0
        self.pxx = pxx
        self.pzx = pzx

        self.xd = self.px0.sample().shape[1]
        self.zd = self.pzx.sample(self.px0.sample()).shape[1]

    def sample(self, T):
        x = torch.empty((T, self.xd))
        z = torch.empty((T, self.zd))
        x[0, :] = self.px0.sample()
        z[0, :] = self.pzx.sample(x[0, :])

        for t in range(1, T):
            x[t, :] = self.pxx.sample(x[t-1, :])
        z = self.pzx.sample(x)

        return x, z

    def predict(self, x):
        return self.pxx.sample(x)

    def observe(self, x, z):
        z_probs = self.pzx.log_prob(z, x).exp()
        z_probs /= z_probs.sum()
        N = x.shape[0]
        cat = Categorical(probs=z_probs)
        x_inds = cat.sample((N,))
        return x[x_inds]

    def pf_step(self, x, z):
        x = self.predict(x)
        return self.observe(x, z)

    def pf(self, z, N):
        T = z.shape[0]
        x = torch.empty((T, N, self.xd))
        x[0, :, :] = self.px0.sample((N,))
        x[0, :, :] = self.observe(x[0, :, :], z[0, :])

        for t in range(1, T):
            x[t, :, :] = self.pf_step(x[t-1, :, :], z[t, :])

        return x

# %%
class MVNCondDist(CondDist):
    def __init__(self, mvn):
        self.mvn = mvn

    def log_prob(self, x, y):
        return self.mvn.log_prob(x - y)

    def sample(self, x, shape=(1,)):
        return x + self.mvn.sample((x.shape[0],))

class MVNDist(CondDist):
    def __init__(self, mvn):
        self.mvn = mvn

    def log_prob(self, x):
        return self.mvn.log_prob(x)

    def sample(self, shape=(1,)):
        return self.mvn.sample(shape)

px0 = MVNDist(MultivariateNormal(
    torch.tensor([0.]), covariance_matrix=torch.tensor([[1.]])))
pxx = MVNCondDist(MultivariateNormal(
    torch.tensor([0.]), covariance_matrix=torch.tensor([[1.]])))
pzx = MVNCondDist(MultivariateNormal(
    torch.tensor([0.0]), covariance_matrix=torch.tensor([[3.]])))

mvn = MultivariateNormal(
    torch.tensor([0.]), covariance_matrix=torch.tensor([[1.]]))
ssm = SSM(px0, pxx, pzx)

# %%
T = 100
N = 100
x, z = ssm.sample(T)
plot(x, z)

x_hat = ssm.pf(z, N)
plot(x, z, x_hat.mean(dim=1).squeeze(), x_hat.std(dim=1).squeeze())

# %%
