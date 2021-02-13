# Generates names from different languages using an RNN.

# Reading material:
# https://karpathy.github.io/2015/05/21/rnn-effectiveness/
# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

# The data loading part of this code was copied and adapted from
# https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

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
class MyRNN(nn.Module):
    def __init__(self, ncats, ntokens, nhidden, nembed, nout):
        super(MyRNN, self).__init__()
        self.cat_embed = nn.Embedding(ncats, nembed)
        self.input_embed = nn.Embedding(ntokens, nembed)
        self.hidden = nn.Linear(nembed + nembed + nhidden, nhidden)
        self.output = nn.Linear(nembed + nhidden, nout)

    def forward(self, cat, input, hidden):
        cat = self.cat_embed(cat)
        input = self.input_embed(input)
        hidden = nn.functional.tanh(self.hidden(torch.cat([cat, input, hidden], dim=1)))
        output = self.output(torch.cat([hidden, input], dim=1))
        return hidden, output

    def get_init_hidden(self, batch_sz):
        return torch.zeros(batch_sz, nhidden)

# %%
ncats = n_categories
ntokens = n_letters
nhidden = 128
nembed = 5
nout = n_letters
model = MyRNN(ncats, ntokens, nhidden, nembed, nout)
model.train()

nsteps = 10000
log_every = 500
batch_sz = 4
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

sum_loss = 0.0
for step in range(nsteps):
    model.zero_grad()
    cat, input, target = make_batch(batch_sz)
    hidden = model.get_init_hidden(batch_sz)
    loss = torch.tensor(0.0)
    for i in range(len(input)):
        hidden, output = model(cat, input[i], hidden)
        loss += nn.functional.cross_entropy(output, target[i])
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()
    if step != 0 and step % log_every == 0:
        print(f'step: {step} / loss: {sum_loss / log_every}')
        sum_loss = 0.0

model.eval()

# %%
def get_sample(model, cat):
    hidden = model.get_init_hidden(1)
    cat = categoryTensor(cat)
    input = torch.tensor(START)
    sample = []

    with torch.no_grad():
        while True:
            hidden, output = model(cat.unsqueeze(0), input.unsqueeze(0), hidden)
            output_dist = nn.functional.softmax(output)[0]
            output = Categorical(output_dist).sample()
            if output == END:
                break

            input = output
            sample.append(output)

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
