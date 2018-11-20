import torch
import torch.nn as nn
import numpy as np


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, device, tie_weights=False):
        super(RNNModel, self).__init__()
        self.device = device
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.inp = nn.Linear(ntoken, nhid)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(nhid, nhid, nlayers, nonlinearity=nonlinearity)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
#         if tie_weights:
#             if nhid != ninp:
#                 raise ValueError('When using the tied flag, nhid must be equal to emsize')
#             self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        
        self.ntoken = ntoken

    def init_weights(self):
        self.inp.bias.data.zero_()
        self.inp.weight.data = torch.randn(self.inp.weight.size()) * 0.1
        #
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.weight_ih_l0.data = (
            torch.randn(self.rnn.weight_ih_l0.size()) * 0.1)
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.weight_hh_l0.data = (
            torch.randn(self.rnn.weight_hh_l0.size()) * 0.1)        
        #
        self.decoder.bias.data.zero_()
        self.decoder.weight.data = (
            torch.randn(self.decoder.weight.size()) * 0.1)

    def forward(self, input, hidden):
        # to oneHot
        inputs = torch.empty(input.shape[0], input.shape[1], self.ntoken).to(self.device)
        for x in range(input.shape[0]):
            for y in range(input.shape[1]):
                inputs[x, y, :] = torch.Tensor(np.arange(self.ntoken)).long().to(self.device) == input[x, y]

        inp = self.inp(inputs)
        output, hidden = self.rnn(inp, hidden)        
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)