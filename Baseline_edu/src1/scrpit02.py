import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import _pickle as pickle

# Generating a noisy multi-sin wave 

def sine_2(X, signal_freq=60.):
    return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def sample(sample_size, dataTensor):
    random_offset = random.randint(0, sample_size)
    X = np.arange(sample_size)
    Y = dataTensor.data[X]
    return Y

# Define the model

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, dictLen):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dictLen = dictLen

        self.inp = nn.Linear(1, hidden_size)
        self.inp.weight.data = torch.randn(self.inp.weight.size()) * 0.1
        
        self.rnn = nn.RNN(hidden_size, hidden_size, 1)
        self.rnn.weight_ih_l0.data = torch.randn(
            self.rnn.weight_ih_l0.size()) * 0.1
        self.rnn.weight_hh_l0.data = torch.randn(
            self.rnn.weight_hh_l0.size()) * 0.1
        
        self.out = nn.Linear(hidden_size, 
                             self.dictLen)        
        self.out.weight.data = torch.randn(self.out.weight.size()) * 0.1
        
        self.softMax = nn.LogSoftmax()

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        output = nn.LogSoftmax(output)
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, self.dictLen))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        outputs = outputs.reshape((-1,))
        return outputs, hidden

myPath = "../dataset/"
with open(myPath + "corpus.pickle", "rb") as f:
    corpus = pickle.load(f)
    
n_epochs = 100
n_iters = 50
hidden_size = 10

model = SimpleRNN(hidden_size, corpus.dictionary.__len__())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

losses = np.zeros(n_epochs) # For plotting

for epoch in range(n_epochs):

    for iter in range(n_iters):
        _inputs = sample(50, corpus.train)
        inputs = Variable(_inputs[:-1].float())
        targets = Variable(_inputs[1:].float())

        # Use teacher forcing 50% of the time
        force = random.random() < 0.5
        outputs, hidden = model(inputs, None, force)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss.item()

    if epoch > 0:
        print(epoch, loss.item())

    # Use some plotting library
    # if epoch % 10 == 0:
        # show_plot('inputs', _inputs, True)
        # show_plot('outputs', outputs.data.view(-1), True)
        # show_plot('losses', losses[:epoch] / n_iters)

        # Generate a test
        # outputs, hidden = model(inputs, False, 50)
        # show_plot('generated', outputs.data.view(-1), True)

# Online training
hidden = None

while True:
    inputs = get_latest_sample()
    outputs, hidden = model(inputs, hidden)

    optimizer.zero_grad()
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()




















