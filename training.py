#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

teacher_force_ratio = 0.5
clip = 5

def training(input_var, target_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    loss = 0
    
    #Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    #Get length
    input_len = input_var.size()[0]
    target_len = target_var.size()[0]
    
    #Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)
    
    #Then go to decoder, prepare input, context, hidden first
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda()
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_context = decoder_context.cuda()
    
    #Training decoder, use teacher enforce
    is_teacher_forcing = random.random() < teacher_force_ratio
    if is_teacher_enforce:
        for i in range(target_len):
            decoder_output, decoder_context, decoder_hidden, decoder_atten = decoder(decoder_input,
                                                                                    decoder_hidden,
                                                                                    decoder_context,
                                                                                    encoder_outputs)
            loss += criterion(decoder_output, target_var[i])
            decoder_input = target_var[i]
    else:
        for i in range(target_len):
            decoder_output, decoder_context, decoder_hidden, decoder_atten = decoder(decoder_input,
                                                                                    decoder_hidden,
                                                                                    decoder_context,
                                                                                    encoder_outputs)
            loss += criterion(decoder_output, target_var[i])
            
            _, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor(ni))
            decoder_input = decoder_input.cuda()
            if ni == EOS_token:
                break
    
    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_len

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))