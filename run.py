#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import torch
from torch import optim
import torch.nn as nn

from models import *
from preprocess import *
from training import *

#Read Data
path = '/home/andy/data/lang_pairs/'
input_lang, output_lang, pairs = prepare_data(path, 'eng', 'deu')

#Set parameter
attn_mode = "general"
hidden_size = 500
n_layers = 2
dropout = 0.05

#initial model
encoder = EncoderRNN(input_lang.nwords, hidden_size, n_layers)
encoder = encoder.cuda()
decoder = AttnDecoderRNN(attn_mode, hidden_size, output_lang.nwords, n_layers, dropout=dropout)
decoder = decoder.cuda()

#optimizor and learning_rate
learning_rate = 1e-4
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# plotting and data
n_epochs = 50000
plot_every = 300
print_every = 1000

#history storage
plot_losses = []
plot_loss_total = 0
print_loss_total = 0

#count time
start = time.time()

for epoch in range(1, n_epochs+1):
    # Get data
    training_pair = variable_pair(random.choice(pairs))
    input_var = training_pair[0]
    output_var = training_pair[1]

    #Running traning
    loss = training(input_var, output_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
#     print(loss)
    #keep track of loss
    plot_loss_total += loss
    print_loss_total += loss
    
    #keep plot
    if epoch % plot_every == 0:
        plot_losses.append(plot_loss_total/plot_every)
        plot_loss_total = 0
    
    #print states
    if epoch % print_every == 0:
        print_loss_ave = print_loss_total / print_every
        print("%s (%d %d%%) %.4f" % (since_time(start, epoch/n_epochs), epoch, epoch/n_epochs, print_loss_ave))
        print_loss_total = 0
        