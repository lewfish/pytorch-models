# Generates names from different languages using a transformer.

# The data loading part of this code was copied and adapted from
# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

# The transformer code was based on:
# http://peterbloem.nl/blog/transformers

# %%
from __future__ import unicode_literals, print_function, division

from IPython import get_ipython
from io import open
import glob
import os
import unicodedata
import string
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

get_ipython().run_line_magic('matplotlib', 'inline')

# %%
all_letters = string.ascii_letters + " .,;'-"
START = len(all_letters)
END = len(all_letters) + 1
n_letters = len(all_letters) + 2

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('/opt/data/pytorch-tutorial-data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))

# %%
# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

def categoryTensor(category):
    return torch.tensor(all_categories.index(category))

def line2tensor(line):
    return torch.tensor([START] + [all_letters.find(letter) for letter in line] + [END])

def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    line_tensor = line2tensor(line)
    input_line_tensor = line_tensor[0:-1]
    target_line_tensor = line_tensor[1:]
    return category_tensor, input_line_tensor, target_line_tensor

def make_batch(batch_sz):
    samples = []
    for i in range(batch_sz):
        samples.append(randomTrainingExample())
    max_len = torch.tensor([len(s[1]) for s in samples]).max()

    batch_cat = torch.cat([s[0].unsqueeze(0) for s in samples])
    batch_input = torch.full((max_len, batch_sz), END, dtype=torch.long)
    batch_target = torch.full((max_len, batch_sz), END, dtype=torch.long)
    for i, s in enumerate(samples):
        batch_input[0:len(s[1]), i] = s[1]
        batch_target[0:len(s[2]), i] = s[2]
    return batch_cat, batch_input, batch_target

# %%
class SelfAttention(nn.Module):
    def __init__(self, k, nheads=1, causal_mask=False):
        super().__init__()

        self.k = k
        self.nheads = nheads
        self.causal_mask = causal_mask

        self.key = nn.Linear(k, k * nheads)
        self.query = nn.Linear(k, k * nheads)
        self.values = nn.Linear(k, k * nheads)
        self.out = nn.Linear(k * nheads, k)

    def forward(self, x):
        # x is [t, b, k]
        t, b = x.shape[0:2]
        h = self.nheads
        # x is [b, t, k]
        x = x.transpose(0, 1).contiguous()

        # [b, t, h * k] -> [b * h, t, k]
        key = self.key(x).view(b, t, h, k).transpose(1, 2).contiguous().view(-1, t, k)
        query = self.query(x).view(b, t, h, k).transpose(1, 2).contiguous().view(-1, t, k)
        values = self.values(x).view(b, t, h, k).transpose(1, 2).contiguous().view(-1, t, k)

        raw_att = torch.bmm(key, query.transpose(1, 2)) / (self.k ** 0.5)
        if self.causal_mask:
            mask_inds = torch.triu_indices(t, t, offset=1)
            raw_att[:, mask_inds[0, :], mask_inds[1, :]] = float('-inf')
        att = nn.functional.softmax(raw_att, dim=2)
        out = torch.bmm(att, values)
        # [b * h, t, k] -> [b, h, t, k] -> [b, t, h, k]
        out = out.view(b, h, t, k).transpose(1, 2)
        # [b, t, h, k] -> [t, b, h, k] -> [t, b, h * k]
        out = out.transpose(0, 1).contiguous().view(t, b, -1)
        out = self.out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, k, nheads=1, hidden_factor=4, causal_mask=False):
        super().__init__()

        self.sa = SelfAttention(k, nheads=nheads, causal_mask=causal_mask)
        self.ln1 = nn.LayerNorm(k)
        self.mlp = nn.Sequential(
            nn.Linear(k, k * hidden_factor),
            nn.ReLU(),
            nn.Linear(k * hidden_factor, k),
        )
        self.ln2 = nn.LayerNorm(k)

    def forward(self, x):
        # x is [t, b, k]
        sa = self.sa(x)
        x = x + sa
        x = self.ln1(x)
        mlp = self.mlp(x)
        x = x + mlp
        x = self.ln2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, ncats, ntokens, nblocks=2, nheads=1, max_len=20, k=32,
                 hidden_factor=4, causal_mask=False):
        super().__init__()

        self.max_len = max_len
        self.pos_embed = nn.Embedding(max_len, k)
        self.cat_embed = nn.Embedding(ncats, k)
        self.token_embed = nn.Embedding(ntokens, k)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(k, nheads, hidden_factor, causal_mask=causal_mask)
            for i in range(nblocks)])
        self.classifier = nn.Linear(k, ntokens)

    def forward(self, cat, input):
        # cat is [b]
        # input is [t, b]
        seq_len = input.shape[0]
        if seq_len > self.max_len:
            raise Exception('Input is longer than max_len')
        batch_sz = input.shape[1]
        pos = torch.arange(0, seq_len).unsqueeze(1).repeat(1, batch_sz) # [t, b]
        cat = cat.unsqueeze(0).repeat(seq_len, 1) # [t, b]
        x = self.pos_embed(pos) + self.cat_embed(cat) + self.token_embed(input)
        x = self.transformer_blocks(x)
        logits = self.classifier(x)
        return logits

# %%
ncats = n_categories
ntokens = n_letters
nblocks = 8
nheads = 8
max_len = 30
k = 128

model = Transformer(
    ncats, ntokens, nblocks=nblocks, nheads=nheads, max_len=max_len, k=k,
    causal_mask=True)
model.train()

nsteps = 10000
log_every = 100
batch_sz = 4
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

sum_loss = 0.0
for step in range(nsteps):
    model.zero_grad()
    cat, input, target = make_batch(batch_sz)
    output = model(cat, input)
    loss = nn.functional.cross_entropy(output.view(-1, ntokens), target.view(-1))
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()
    if step != 0 and step % log_every == 0:
        print(f'step: {step} / loss: {sum_loss / log_every}')
        sum_loss = 0.0

model = model.eval()

# %%
def get_sample(model, cat):
    cat = categoryTensor(cat)
    input = [START]

    with torch.no_grad():
        while True:
            output = model(cat.unsqueeze(0), torch.tensor(input).unsqueeze(1))
            output_dist = nn.functional.softmax(output[-1, 0, :])
            sample = input[1:]

            output = Categorical(output_dist).sample()
            if output == END:
                break

            input.append(output)

    sample = input[1:]
    return ''.join([all_letters[s.item()] for s in sample])

def get_samples(model, cat, nsamples):
    return [get_sample(model, cat) for i in range(nsamples)]

def print_samples():
    for cat in all_categories:
        samples = get_samples(model, cat, 10)
        print(cat)
        print(samples)
        print()

print_samples()

# %%
