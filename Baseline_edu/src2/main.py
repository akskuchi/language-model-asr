# coding: utf-8
import argparse
import time
import numpy as np
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

import data
import model

from collections import Counter
from nlgeval import NLGEval

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings (if 0: OneHot instead of embeddings)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--shortlist', type=int, help='How many words to include in the shortlist (-1 for no shortlist)', default=None)
parser.add_argument('--unk-token', type=str, help='Word for unknown token in corpus', default='<unk>')
parser.add_argument('--metrics-k', type=int, help='How many words to predict for metrics', default=3)
parser.add_argument('--show-predictions-during-evaluation', action='store_true',
                    help='Whether to show predicted sentences during evaluation')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=None,
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--save-statistics', type=str, default=None)
parser.add_argument('--initialization', type=str, default="rand",
                    help='"rand" (var=0.1), "xavier", "Kaiming"')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

print("MODEL: Loading corpus")
corpus = data.Corpus(args.data, shortlist=args.shortlist, unk_token=args.unk_token)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# â”Œ a g m s â”�
# â”‚ b h n t â”‚
# â”‚ c i o u â”‚
# â”‚ d j p v â”‚
# â”‚ e k q w â”‚
# â”” f l r x â”˜.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens_out = len(corpus.dictionary)
ntokens_in = len(corpus.dictionary.idx2word)
model = model.RNNModel(args.model, ntokens_in, ntokens_out, args.emsize, args.nhid,
                       args.nlayers, device, args.tied, args.dropout, args.initialization).to(device)

criterion = nn.CrossEntropyLoss()


def batch_to_word_sequence(data):
    """Convert an input batch into a sequence of words and word indices."""
    data_npy = data.cpu().detach().numpy()

    return [
        [corpus.dictionary.idx2word[w] for w in s]
        for s in data_npy.T
    ], data_npy.T.tolist()


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# â”Œ a g m s â”� â”Œ b h n t â”�
# â”” b h n t â”˜ â”” c i o u â”˜
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


print("MODEL: Creating evaluator")
EVALUATOR = NLGEval(no_glove=True, no_skipthoughts=True)


def average_statistics(statistics):
    avg_statistics = {k: 0 for k in statistics[0]}

    for statistic in statistics:
        for key in avg_statistics.keys():
            avg_statistics[key] += statistic[key] / len(statistics)

    return avg_statistics


def output_to_preds(output):
    return F.softmax(output.view(-1, len(corpus.dictionary)), dim=1).argmax(1).detach().flatten().cpu().numpy()


def predict_n(model, n, hidden, last):
    for i in range(n):
        output, hidden = model(torch.tensor(np.array([[last]]).T).to(device), hidden)
        last = output_to_preds(output)[0]
        yield corpus.dictionary.reverse_shortlist_mapping[last]


def generate_metrics(data, model, k):
    char_sequences, encoding_sequences = batch_to_word_sequence(data)

    for char_seq, enc_seq in zip(char_sequences, encoding_sequences):
        if len(enc_seq) <= k:
            continue

        start_char_seq, start_enc_seq = char_seq[:-k], enc_seq[:-k]
        hidden = model.init_hidden(1)

        for enc in start_enc_seq:
            output, hidden = model(torch.tensor(np.array([[enc]]).T).to(device), hidden)

        next_1 = output_to_preds(output)[0]

        # Now predict the next n - 1
        next_enc_n = list(predict_n(model, k - 1, hidden, next_1))
        encodings = start_enc_seq + [next_1] + next_enc_n

        real_sentence = " ".join(char_seq)
        predicted_seq = [corpus.dictionary.idx2word[w] for w in encodings]
        predicted_sentence = " ".join(predicted_seq)

        accuracy = len([1 for r, p in zip(char_seq, predicted_seq) if r == p]) / len(char_seq)

        if args.show_predictions_during_evaluation:
            print("REFERENCE:", "".join(real_sentence))
            print("PREDICTED:", "".join(predicted_sentence))
        metrics = EVALUATOR.compute_individual_metrics(ref=[" ".join(real_sentence)],
                                                       hyp=" ".join(predicted_sentence))
        metrics["accuracy"] = accuracy
        yield metrics


def repackage_with_shortlist(targets, dictionary):
    """Given some targets array, move it back to the CPU shortistify it.

    This should have no effect if shortlists are disabled.
    """
    targets_array = targets.detach().cpu().numpy()
    repackaged = np.array([dictionary.shortlist_mapping[w] for w in targets_array.flatten()]).reshape(targets_array.shape)
    return torch.tensor(repackaged).long().to(device)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens_out = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    recorded_metrics = []
    with torch.no_grad():
        # Need this otherwise pytorch runs out of memory
        for i in range(0, min(data_source.size(0) - 1, 1000), args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens_out)
            targets = repackage_with_shortlist(targets, corpus.dictionary)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

            recorded_metrics.extend(list(generate_metrics(data, model, args.metrics_k)))
    return total_loss / (len(data_source) - 1), average_statistics(recorded_metrics)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens_out = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        targets = repackage_with_shortlist(targets, corpus.dictionary)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens_out), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def format_statistics(statistics):
    return '| '.join([
        '{0}: {1:.2f}'.format(k, s) for k, s in statistics.items()
    ])


# Loop over epochs.
lr = args.lr
best_val_loss = None

collected_statistics = pd.DataFrame()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss, statistics = evaluate(val_data)

        # Append statistics here, we will dump them later
        statistics['val_loss'] = val_loss
        statistics['ppl'] = math.exp(val_loss)
        collected_statistics = collected_statistics.append(statistics, ignore_index=True)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | {}'.format(epoch, (time.time() - epoch_start_time),
                                                val_loss, math.exp(val_loss), format_statistics(statistics)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if args.save:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model if we were saving them.
if args.save:
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

# Run on test data.
test_loss, statistics = evaluate(test_data)
statistics['val_loss'] = test_loss
statistics['ppl'] = math.exp(test_loss)
collected_statistics = collected_statistics.append(statistics, ignore_index=True)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} {}'.format(
    test_loss, math.exp(test_loss), format_statistics(statistics)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

if args.save_statistics:
    collected_statistics.to_csv(args.save_statistics)
