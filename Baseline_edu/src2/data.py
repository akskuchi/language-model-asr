import os
import torch
from collections import Counter, defaultdict

class Dictionary(object):
    def __init__(self, unk_token):
        # Explicitly create a mapping for UNK, since we'll
        # be relying on it later
        self.word2idx = {
            unk_token: 0
        }
        self.idx2word = [unk_token]
        self.word_frequencies = Counter()
        self.unk_token = unk_token

        # Not known -> UNK
        self.shortlist_mapping = defaultdict(lambda: 0)
        self.reverse_shortlist_mapping = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        self.word_frequencies[word] += 1
        return self.word2idx[word]

    def create_shortlist_mapping(self, length):
        self.shortlist_mapping[self.word2idx[self.unk_token]] = len(self.shortlist_mapping)
        self.reverse_shortlist_mapping.append(self.word2idx[self.unk_token])
        for word, freq in self.word_frequencies.most_common(None if length == -1 else length):
            self.shortlist_mapping[self.word2idx[word]] = len(self.shortlist_mapping)
            self.reverse_shortlist_mapping.append(self.word2idx[word])

    def __len__(self):
        return len(self.reverse_shortlist_mapping)


class Corpus(object):
    def __init__(self, path, shortlist=None, unk_token='<unk>'):
        self.dictionary = Dictionary(unk_token)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))

        # Create the shortlist mapping
        self.dictionary.create_shortlist_mapping(shortlist)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
