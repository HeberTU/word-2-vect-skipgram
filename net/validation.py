# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:39:27 2020

@author: HTRUJILLO
"""
import numpy as np
import random
import torch

def cosine_similarity(embedding, valid_size = 16, valid_window = 100, device = 'cpu'):
    
    
    embed_vectors = embedding.weight
    
    magnitudes = embed_vectors.pow(2).sum(dim = 1).sqrt().unssqueeze(0)
    
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities