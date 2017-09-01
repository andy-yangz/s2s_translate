#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embeded = self.embedding(words_input).view(seq_len, 1, -1)
        output, hidden = self.gru(embeded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        hidden = hidden.cuda()
        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size*2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len)).cuda()
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            return hidden.squeeze().dot(encoder_output.squeeze())
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze().dot(energy.squeeze())
            return  energy
        if self.method == 'concat':
            energy = self.attn(torch.cat(hidden, encoder_output), 1)
            energy = self.other.dot(energy.squeeze())
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_mode, hidden_size, output_size, n_layers=1, dropout=0.1):
        super().__init__()
        # Parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, 
                          dropout=dropout)
        self.out = nn.Linear(hidden_size*2, output_size)
        
        #Attn mode
        if attn_mode != 'none':
            self.attn = Attn(attn_mode, hidden_size)
            
    def forward(self, word_input, last_hidden, last_context, encoder_outputs):
        word_embeded = self.embedding(word_input).view(1, 1, -1)
        
        rnn_input = torch.cat((word_embeded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        
        attn_weight = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weight.bmm(encoder_outputs.transpose(0, 1))
        
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, context, hidden, attn_weight


if __name__ == '__main__':
    # Test models
    encoder_test = EncoderRNN(10, 10, 2).cuda()
    decoder_test = AttnDecoderRNN('general', 10, 10, 2).cuda()
    print(encoder_test)
    print(decoder_test)

    encoder_hidden = encoder_test.ini_hidden()
    word_input = Variable(torch.LongTensor([1, 2, 3])).cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

    decoder_attns = torch.zeros(1, 3, 3)
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size)).cuda()

    for i in range(3):
        decoder_output, decoder_context, decoder_hidden, decoder_attn \
            = decoder_test(word_input[i], decoder_hidden, decoder_context, 
                           encoder_outputs)
        
        print(decoder_output.size(), 
              decoder_hidden.size(), 
              decoder_attn.size())
        decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data