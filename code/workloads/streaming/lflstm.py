import sys
TOOLKIT = "../CMU-MultimodalSDK/"
sys.path.append(TOOLKIT)

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
from collections import defaultdict

# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
EPS = 0

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']


def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths

def multi_collate_sample(batch, n_frames):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.FloatTensor(sample[0][0][:n_frames]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1][:n_frames]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2][:n_frames]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0][:n_frames].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths

class LFLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, fc1_size, output_size, dropout_rate):
        super(LFLSTM, self).__init__()
        self.input_size = input_sizes
        self.hidden_size = hidden_sizes
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.embed = nn.Embedding(len(word2id), input_sizes[0])
        self.trnn1 = nn.LSTM(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = nn.LSTM(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = nn.LSTM(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = nn.LSTM(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = nn.LSTM(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = nn.LSTM(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.fc1 = nn.Linear(sum(hidden_sizes)*4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        self.bn = nn.BatchNorm1d(sum(hidden_sizes)*4)

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths.tolist())
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.tolist())
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

        
    def fusion(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        #sentences = self.embed(sentences)
        
        # extract features from text modality
        final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        
        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)

        
        # simple late fusion -- concatenation + normalization
        h = torch.cat((final_h1t, final_h2t, final_h1v, final_h2v, final_h1a, final_h2a),
                       dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def forward(self, sentences, visual, acoustic, lengths):
        batch_size = lengths.size(0)
        h = self.fusion(sentences, visual, acoustic, lengths)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o

def make_model(hidden, fc, dropout):
    text_size = 300
    visual_size = 35
    acoustic_size = 74
    output_size = 7
    input_sizes = [text_size, visual_size, acoustic_size]
    
    hidden_sizes = [int(text_size * hidden), int(visual_size * hidden), int(acoustic_size * hidden)]
    fc1_size = int(sum(hidden_sizes) // fc)
    
    return LFLSTM(input_sizes, hidden_sizes, fc1_size, output_size, dropout)