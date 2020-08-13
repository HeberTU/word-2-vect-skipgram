# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:01:36 2020

@author: HTRUJILLO
"""
import torch
import torch.optim as optim
from torch import nn
from utilities.data_prep import get_batches
from net.validation import cosine_similarity

def train(model, train_words, int_to_vocab, embedding_dim = 300, print_every = 500, epochs = 5):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("The model will Train on {}".format(device))
    
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    steps = 0
    
    ep = 0
    for e in range(epochs):
        ep = ep + 1
        for inputs, targets in get_batches(train_words, 128):
            steps += 1
            inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            
            log_ps = model(inputs)
            loss = criterion(log_ps, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if steps % print_every == 0:
                
                valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)
                 
                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                 
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...")
                print("epoch {} de {}".format(ep,epochs))

