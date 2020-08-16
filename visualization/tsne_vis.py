# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 23:03:59 2020

@author: heber
"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_vis(model, int_to_vocab, vis_words = 100):

    embeddings = model.embed.weight.to('cpu').data.numpy()
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:vis_words, :])
    
    fig, ax = plt.subplots(figsize=(16,16))
    for idx in range(vis_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)