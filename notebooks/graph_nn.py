# Classify nodes in a graph using graph neural networks on the CORA dataset
# Some code adapted from https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
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
import networkx as nx
from torch_geometric.datasets import KarateClub
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
kc = KarateClub()[0]
kcnx = to_networkx(kc, to_undirected=True)
nx.draw_networkx(
    kcnx, pos=nx.spring_layout(kcnx, seed=43), with_labels=False,
    node_color=kc.y, cmap="Set2")

# %%
edge_index = torch.LongTensor([
    [0, 0, 2],
    [1, 2, 2]])
vals = torch.ones(3)
x = torch.ones(3, 2)
nb_nodes = x.shape[0]
data = Data(x=x, edge_index=edge_index)
nx.draw_networkx(to_networkx(data))

# %%
sp_index = torch.sparse.FloatTensor(edge_index, vals, torch.Size([3, 3]))
dense_index = sp_index.to_dense()
X = torch.arange(0, 9).view(3, 3).float()
X2 = dense_index.transpose(0, 1).mm(X)
X3 = sp_index.transpose(0, 1).mm(X)
assert torch.equal(X2, X3)

# %%
def add_self_edges(edge_index, nb_nodes):
    node_inds = torch.arange(0, nb_nodes)
    self_edge_index = torch.cat([node_inds.unsqueeze(0), node_inds.unsqueeze(0)])
    return torch.cat([edge_index, self_edge_index], dim=1)

class MyGCNConv(nn.Module):
    def __init__(self, in_sz, out_sz):
        super().__init__()
        self.fc = nn.Linear(in_sz, out_sz)

    def forward(self, X, edge_index):
        # Use sparse matrix multiplication to sum the neighbors of each node.
        sum_neighbors = edge_index.transpose(0, 1).mm(X)
        return self.fc(sum_neighbors)

class MyGCN(nn.Module):
    def __init__(self, in_sz, hidden_sz, out_sz, ):
        super().__init__()
        self.conv1 = MyGCNConv(in_sz, hidden_sz)
        self.conv2 = MyGCNConv(hidden_sz, out_sz)
        self.edge_index = None

    def get_sparse_index(self, x, edge_index):
        nb_nodes, nb_features = x.shape
        edge_index = add_self_edges(edge_index, nb_nodes)
        node_degs = torch.bincount(edge_index[1, :], minlength=nb_nodes).float()
        deg_i = node_degs[edge_index[0, :]]
        deg_j = node_degs[edge_index[1, :]]
        norms = 1.0 / torch.sqrt(deg_i * deg_j)
        sp_index = torch.sparse.FloatTensor(
            edge_index, norms, torch.Size([nb_nodes, nb_nodes]))
        return sp_index

    def forward(self, x, edge_index):
        if self.edge_index is None:
            self.edge_index = self.get_sparse_index(x, edge_index)
        x = self.conv1(x, self.edge_index)
        x = x.relu()
        x = self.conv2(x, self.edge_index)
        return x

# %%
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# %%
def train_model(model, nb_epochs=200, use_edge_index=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index) if use_edge_index else model(data.x)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def test():
        model.eval()
        out = model(data.x, data.edge_index) if use_edge_index else model(data.x)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc

    for epoch in range(1, nb_epochs+1):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

# %%
hidden_sz = 16
mlp = nn.Sequential(
    nn.Linear(data.num_features, hidden_sz),
    nn.ReLU(),
    nn.Linear(hidden_sz, dataset.num_classes)
)
train_model(mlp)

# %%
hidden_sz = 16
nb_epochs = 100
gcn = MyGCN(data.num_features, hidden_sz, dataset.num_classes)
train_model(gcn, nb_epochs=nb_epochs, use_edge_index=True)
