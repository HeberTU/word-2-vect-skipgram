# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:08:50 2020

@author: HTRUJILLO
"""
import os 
os.chdir("C:/Users/htrujillo/projects/Implementing_Word2Vec_SkipGram")

from utilities.data_prep import load_data, preprocess,  create_lookup_tables, subsampling
from net.architecture import SkipGram
from net.train import train

text = load_data("data/text8")

words = preprocess(text)

print("Total words in text: {}".format(len(words)))
print("Unique words: {}".format(len(set(words))))

vocab_to_int, int_to_vocab = create_lookup_tables(words)

int_words = [vocab_to_int[word] for word in words]

print(int_words[:30])

train_words = subsampling(int_words, threshold = 1e-5)

print(train_words[:30])

embedding_dim = 300


model = SkipGram(len(vocab_to_int), embedding_dim)


train(model, train_words, int_to_vocab, embedding_dim = 300, print_every = 100, epochs = 5)
