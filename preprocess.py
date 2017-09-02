#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import unicodedata
import re
import os

from torch.autograd import Variable

# Special token
SOS_token = 0
EOS_token = 1

# Max length, and good prefixs to filter pairs
MAX_LENGTH = 20
good_prefixes = ('i', 'you', 'he', 'she', 'we', 'they')

class Lang:
    """Language class. 
    Create several information about input corpus."""
    def __init__(self, name):
        self.name = name
        self.word2ind = {'SOS':0, 'EOS':1}
        self.word2cnt = {}
        self.ind2word = {0:'SOS', 1:'EOS'}
        self.nwords = 2
    
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word.lower())
    
    def index_word(self, word):
        if word not in self.word2ind:
            self.word2ind[word] = self.nwords
            self.word2cnt[word] = 1
            self.ind2word[self.nwords] = word
            self.nwords += 1
        else:
            self.word2cnt[word] += 1
            
    def __str__(self):
        return 'This is %s, it has %d words' % (self.name, self.nwords)


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z0-9.!?\t]', r' ', s)
    return s


def read_langs(path, lang1_n, lang2_n):
    lang1 = Lang(lang1_n)
    lang2 = Lang(lang2_n)
    data = open(os.path.join(path,'%s-%s.txt'%(lang1_n, lang2_n)))
    pairs = []

    for line in data:
        pair = normalize_string(line).strip().split('\t')
        pairs.append(pair)
        
    return lang1, lang2, pairs


def filter(p):
    return(len(p[0]) < MAX_LENGTH and p[0].startswith(good_prefixes))

def filter_pairs(pairs):
    return[pair for pair in pairs if filter(pair)]

def prepare_data(path, lang1_n, lang2_n):
    input_lang, output_lang, pairs = read_langs(path, lang1_n, lang2_n)
    print("Read %d sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimming to %d pairs." % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
    return input_lang, output_lang, pairs

def indexes_from_sentence(lang, sentence):
    return [lang.word2ind[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    var = var.cuda()
    return var

def variables_from_pair(pair, input_lang, output_lang):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)