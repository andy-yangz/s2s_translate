#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import random
import torch
from torch import optim
import torch.nn as nn

from models import *
from preprocess import *
from training import *
from util import *

#Add argparse part
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--evaluate', help="Evaluate our model",
                    action="store_true" )
parser.add_argument('-w', '--w2v_path', help='Load pretrained word2vec')
parser.add_argument('-t', '--test', help='Use sampel of data to test model',
                    action='store_true')
parser.add_argument('-s', '--save', help='Path to save trained model, and data.',
                    default='checkpoint.pth.tar')
args = parser.parse_args()


#Read Data
if args.evaluate:
    print("Evaluation Mode.")
    check_point = torch.load('checkpoint.pth.tar')

    #Load language information
    input_lang = check_point['input_lang']
    output_lang = check_point['output_lang']
    pairs = check_point['pairs']
else:    
    path = '/home/andy/data/lang_pairs/'
    if args.test:
        input_lang, output_lang, pairs = prepare_data(path, 'eng', 'deu_test')
    else:
        input_lang, output_lang, pairs = prepare_data(path, 'eng', 'deu')

#Set parameter
attn_mode = "general"
hidden_size = 500
n_layers = 2
dropout = 0.05

#Load pretrained embed matrix
if args.w2v_path: # Use pretrained w2v
    embed_mat, embed_size = get_embedding_data(args.w2v_path, input_lang)
else: # embed_size euqal hidden size, without pretrained
    embed_mat = None
    embed_size = hidden_size

#initial model
encoder = EncoderRNN(input_lang.nwords, embed_size, hidden_size, n_layers, bidirectional=True)
if args.w2v_path:
    encoder.load_pretrained_embed(pretrained_weights)
encoder = encoder.cuda()
decoder = AttnDecoderRNN(attn_mode, hidden_size, output_lang.nwords, n_layers, dropout_p=dropout)
decoder = decoder.cuda()

if args.evaluate:
    encoder.load_state_dict(check_point['encoder'])
    decoder.load_state_dict(check_point['decoder'])

    evaluate_randomly(pairs, input_lang, output_lang, encoder, decoder)

#optimizor and learning_rate
learning_rate = 1e-4
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# plotting and data
if args.test:
    n_epochs = 50
    plot_every = 2
    print_every = 10
else:
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
    training_pair = variables_from_pair(random.choice(pairs), input_lang, output_lang)
    input_var = training_pair[0]
    output_var = training_pair[1]

    #Running traning
    loss = train(input_var, output_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
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


# Save trained data
save_checkpoint({
    'plot_losses': plot_losses,
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'encoder_optimizer': encoder_optimizer.state_dict(),
    'decoder_optimizer': decoder_optimizer.state_dict(),
    'epoch': epoch + 1,
    'input_lang': input_lang,
    'output_lang': output_lang,
    'pairs': pairs
}, filename=args.save)