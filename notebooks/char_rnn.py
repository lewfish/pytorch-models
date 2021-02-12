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
EOS = len(all_letters)
n_letters = len(all_letters) + 1 # Plus EOS marker

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

def inputTensor(line):
    return torch.tensor([all_letters.find(letter) for letter in line])

def targetTensor(line):
    # second letter to EOS
    return torch.cat([inputTensor(line[1:]), torch.tensor([EOS])])

def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

# %%
class MyRNN(nn.Module):
    def __init__(self, ncats, ninput, nhidden, nembed, nout):
        super(MyRNN, self).__init__()
        self.cat_embed = nn.Embedding(ncats, nembed)
        self.input_embed = nn.Embedding(ninput, nembed)
        self.hidden = nn.Linear(nembed + nembed + nhidden, nhidden)
        self.output = nn.Linear(nembed + nhidden, nout)

    def forward(self, cat, input, hidden):
        cat = self.cat_embed(cat)
        input = self.input_embed(input)
        hidden = nn.functional.tanh(self.hidden(torch.cat([cat, input, hidden])))
        output = self.output(torch.cat([hidden, input]))
        return hidden, output

    def get_init_hidden(self):
        return torch.zeros(nhidden)

# %%
ncats = n_categories
ninput = n_letters
nhidden = 128
nembed = 5
nout = n_letters
model = MyRNN(ncats, ninput, nhidden, nembed, nout)
model.train()

nsteps = 50000
log_every = 1000
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

sum_loss = 0.0
for step in range(nsteps):
    model.zero_grad()
    cat, input, target = randomTrainingExample()
    hidden = model.get_init_hidden()
    loss = torch.tensor(0.0)
    for i in range(len(input)):
        hidden, output = model(cat, input[i], hidden)
        loss += nn.functional.cross_entropy(output.unsqueeze(0), target[i].unsqueeze(0))
    loss.backward()
    optimizer.step()

    sum_loss += loss.item()
    if step != 0 and step % log_every == 0:
        print(f'step: {step} / loss: {sum_loss / log_every}')
        sum_loss = 0.0

model.eval()

# %%
def get_sample(model, cat, start_char):
    hidden = model.get_init_hidden()
    cat = categoryTensor(cat)
    input = inputTensor(start_char)[0]
    sample = [input]

    with torch.no_grad():
        while True:
            hidden, output = model(cat, input, hidden)
            output_dist = nn.functional.softmax(output)
            output = Categorical(output_dist).sample()
            if output == EOS:
                break

            input = output
            sample.append(output)

    return ''.join([all_letters[s.item()] for s in sample])

def get_samples(model, cat, nsamples):
    return [
        get_sample(model, cat, random.choice(string.ascii_uppercase))
        for i in range(nsamples)]

def print_samples():
    for cat in all_categories:
        samples = get_samples(model, cat, 10)
        print(cat)
        print(samples)
        print()

print_samples()
