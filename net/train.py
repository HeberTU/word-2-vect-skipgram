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
from net.architecture import SkipGram

def train(model, train_words, int_to_vocab,batch_size = 128 , embedding_dim = 300, print_every = 500, epochs = 5, verbose = True):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("The model will Train on {}".format(device))
    
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    steps = 0
    
    ep = 0
    for e in range(epochs):
        ep = ep + 1
        
        for inputs, targets in get_batches(train_words, batch_size):
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
                if verbose:
                    print("epoch {} de {}".format(ep,epochs))
                    print("Current Loss {}".format(loss.item()))


def save_embbedings(model , n_vocab, n_embed, path = "skipgram_embed.pth"):
    
    check_point = {
        'state_dict': model.state_dict(),
        'n_vocab': n_vocab,
        'n_embed': n_embed}
    
    torch.save(check_point, path)
    print("Model saved as {}".format(path))
    
  


def load_embbedings(file_path):
    check_point = torch.load(file_path)    
    model = SkipGram(n_vocab = check_point['n_vocab'], n_embed = check_point['n_embed'])
    model.load_state_dict(check_point['state_dict'])
    return model